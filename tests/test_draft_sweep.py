import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from workloads.serving import draft_sweep as sweep


def test_worker_entry_point_runs_from_staged_directory():
    staged = Path(__file__).resolve().parents[1]
    body = sweep.payload("train", sweep.TRAINER, staged)
    _, command = body["script"].split(" && ", 1)
    _, *worker_args = shlex.split(command)
    subprocess.run([sys.executable, *worker_args, "--help"], cwd=staged, check=True,
                   stdout=subprocess.DEVNULL)


def test_sweep_has_matched_one_gpu_evaluations(monkeypatch):
    monkeypatch.setattr(sweep.c, "HORIZONS", sweep.HORIZONS)
    arms = {(exp.arm, exp.mode, exp.draft_tokens) for exp in sweep.experiment_configs}
    assert len(arms) == len(sweep.experiment_configs) == 12
    for mode in sweep.c.GENERATION:
        assert ("ar", mode, 0) in arms
        assert ("sps", mode, 0) in arms
        for draft in (4, 8):
            for arm in ("sps_mtp", "eagle3"):
                assert (arm, mode, draft) in arms
    for exp in sweep.experiment_configs:
        args = sweep.engine_arguments(exp, "/test/hf", sweep.c.EAGLE_REVISION)
        assert args["max_num_seqs"] == 1
        assert (args.get("speculative_config") or {}).get(
            "num_speculative_tokens", 0
        ) == exp.draft_tokens
        if exp.arm.startswith("sps"):
            assert args["model"] == (
                "/test/hf-draft-4" if exp.draft_tokens == 4 else "/test/hf"
            )
        assert sweep.payload("eval", exp, sweep.ROOT)["instance_type"] == "a100.1gpu"


def test_short_export_reuses_weights_and_preserves_slot_prefix(tmp_path):
    import torch
    from safetensors.torch import load_file, save_file

    export = tmp_path / "hf"
    export.mkdir()
    weights = export / "model-00001.safetensors"
    weights.write_bytes(b"original")
    key = "sps_prediction_embeddings"
    embeddings = torch.arange(9 * 2, dtype=torch.float32).reshape(9, 2)
    save_file({key: embeddings}, export / "sps.safetensors")
    (export / "config.json").write_text(json.dumps({"sps_horizons": 9}))
    index = export / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {"total_size": 100}, "weight_map": {}}))

    with pytest.raises(RuntimeError, match="Missing"):
        sweep.prepare_short_export(export, create=False)
    sweep.prepare_short_export(export)
    sweep.prepare_short_export(export, create=False)

    short = tmp_path / "hf-draft-4"
    assert json.loads((short / "config.json").read_text())["sps_horizons"] == 5
    assert (short / weights.name).resolve() == weights
    assert load_file(short / "sps.safetensors")[key].equal(embeddings[:5])
    assert json.loads((short / index.name).read_text())["metadata"]["total_size"] == 68
    save_file({key: embeddings[4:]}, short / "sps.safetensors")
    with pytest.raises(RuntimeError, match="differs"):
        sweep.prepare_short_export(export)


def test_counter_metrics_uses_actual_proposals_and_rounds():
    values = {
        "num_drafts": 10,
        "num_draft_tokens": 38,
        "num_accepted_tokens": 14,
        "num_accepted_tokens_per_pos": [7, 5, 2, 0],
    }
    result = {
        "acceptance_counters_including_warmup": [
            {"name": f"vllm:spec_decode_{key}", "value": value}
            for key, value in values.items()
        ],
    }
    warmup_values = {
        "num_drafts": 2,
        "num_draft_tokens": 8,
        "num_accepted_tokens": 4,
        "num_accepted_tokens_per_pos": [2, 2, 0, 0],
    }
    warmup = [
        {"name": f"vllm:spec_decode_{key}", "value": value}
        for key, value in warmup_values.items()
    ]
    metrics = sweep.counter_metrics(result, 4, warmup)
    assert metrics["tau"] == pytest.approx(2.25)
    assert metrics["acceptance_rate"] == pytest.approx(10 / 30)
    assert metrics["acceptance_by_position"] == [5 / 8, 3 / 8, 2 / 8, 0]
    with pytest.raises(ValueError, match="Inconsistent"):
        sweep.counter_metrics(result, 8, warmup)


@pytest.mark.parametrize("arm", ["eagle3", "sps_mtp"])
def test_snapshot_excludes_warmup(tmp_path, monkeypatch, arm):
    import sys
    from types import SimpleNamespace

    class Engine:
        requests = 0

        def generate(self):
            self.requests += 1

        def get_metrics(self):
            return [
                SimpleNamespace(name="vllm:spec_decode_num_drafts", value=self.requests)
            ]

    vllm = SimpleNamespace(LLM=Engine)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setattr(sweep.bench, "root_for", lambda _: tmp_path)
    untimed = sweep.c.WARMUP_REQUESTS

    def run_graph(*_):
        llm = vllm.LLM()
        for _ in range(untimed + 3):
            llm.generate()

    monkeypatch.setattr(sweep, "graph_job", run_graph)
    exp = next(x for x in sweep.experiment_configs if x.arm == arm)
    sweep.evaluate(exp)
    assert sweep.untimed_counters(exp) == [
        {"name": "vllm:spec_decode_num_drafts", "value": untimed}
    ]


def test_embedding_phase_enables_decoder_checkpoint_recomputation():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from modeling.models.sps.pretrained import PretrainedSPS
    from workloads.serving import draft_sweep_train, vllm_sps_train
    from workloads.serving.vllm_sps_model import QwenSPSMTP, horizon_targets

    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        attention_dropout=0.0,
    )
    parent = PretrainedSPS(Qwen3ForCausalLM(config).eval())
    model = QwenSPSMTP(parent, draft_sweep_train.HORIZONS)
    layer = model.backbone.model.layers[0]
    assert not layer.training

    draft_sweep_train.training_optimizer(vllm_sps_train.optimizer_for, model, True)
    assert layer.training and layer.gradient_checkpointing
    tokens = torch.arange(1, 13)[None]
    labels = horizon_targets(tokens, torch.ones_like(tokens), model.horizons)
    loss, _ = model(tokens[:, :-1], targets=labels)
    loss.backward()
    assert torch.count_nonzero(model.new_embeddings.grad[-1]) > 0
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_greedy_mismatch_is_reported_but_missing_graph_evidence_blocks(
    tmp_path, monkeypatch
):
    ref = {"prompt_token_ids": [1], "expected_token_ids": [2]}
    monkeypatch.setattr(sweep, "BASE_REFERENCE_TOKENS", lambda _, **kwargs: [ref])
    monkeypatch.setattr(sweep.bench, "root_for", lambda _: tmp_path)
    monkeypatch.setattr(sweep.graph, "protocol_signature", sweep.protocol_signature)
    signature = sweep.protocol_signature()
    for arm in sweep.AUDIT_ARMS:
        actual = [3] if arm == "sps" else [2]
        (tmp_path / f"{arm}.json").write_text(
            json.dumps(
                {
                    "protocol_signature": signature,
                    "samples": [
                        dict(ref, actual_token_ids=actual, equal=actual == [2])
                    ],
                }
            )
        )
        (tmp_path / f"graphs-{arm}.json").write_text(
            json.dumps(
                {
                    "protocol_signature": signature,
                    "workers": [
                        {"captured": 1, "target_replays": 1, "mode": "FULL_DECODE_ONLY"}
                    ],
                }
            )
        )
    diagnostics = sweep.audit_evidence({})
    assert diagnostics["sps"]["exact_vs_native"] == 0
    assert diagnostics["sps_mtp-4"]["exact_vs_native"] == 1
    assert diagnostics["sps_mtp-4"]["exact_vs_vllm_sps"] == 0
    path = tmp_path / "graphs-sps_mtp-4.json"
    path.unlink()
    with pytest.raises(FileNotFoundError):
        sweep.audit_evidence({})
