import json

import pytest

from scripts import draft_sweep as sweep


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

    sweep.prepare_short_export(export)
    sweep.prepare_short_export(export)

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
        "num_accepted_tokens_per_pos": [7, 4, 3, 0],
    }
    result = {
        "acceptance_counters_including_warmup": [
            {"name": f"vllm:spec_decode_{key}", "value": value}
            for key, value in values.items()
        ],
    }
    metrics = sweep.counter_metrics(result, 4)
    assert metrics["tau"] == pytest.approx(2.4)
    assert metrics["acceptance_rate"] == pytest.approx(14 / 38)
    assert metrics["acceptance_by_position"] == [0.7, 0.4, 0.3, 0]
    with pytest.raises(ValueError, match="Inconsistent"):
        sweep.counter_metrics(result, 8)
