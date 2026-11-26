"""Isolation and model identity in the four-arm vLLM job design."""

import json
from pathlib import Path

import pytest
from workloads.serving.vllm_sps_evaluate import engine_arguments, select_examples

from experiments import vllm_sps_config as c
from workloads.serving import serving_benchmark, vllm_sps_run
from workloads.serving import vllm_sps_common as common


def test_eval_jobs_are_single_gpu_and_ordinary_sps_shares_speculative_checkpoint():
    exports = "/immutable/final/hf"
    for exp in c.collect_experiments("eval"):
        body = serving_benchmark.payload(exp, Path("/staged"), "eval")
        assert body["instance_type"] == "a100.1gpu"
        args = engine_arguments(exp, exports, "eagle-immutable-revision")
        assert args["max_num_seqs"] == args["tensor_parallel_size"] == 1
        assert not args["enable_prefix_caching"]
        assert not args["async_scheduling"]
        assert args["model"] == (exports if exp.arm.startswith("sps") else c.family_config.MODEL)
        if exp.arm == "sps":
            assert "speculative_config" not in args
        if exp.arm == "eagle3":
            assert args["speculative_config"]["revision"] == "eagle-immutable-revision"


@pytest.mark.parametrize("phase", ["probe", "train"])
def test_training_jobs_use_the_training_worker_and_allocation(phase):
    exp, = c.collect_experiments(phase)
    body = vllm_sps_run.payload(phase, exp, Path("/staged"), "author", "nick")
    assert body["instance_type"] == vllm_sps_run.launch.INSTANCE_TYPES_BY_NUM_GPUS[exp.num_gpus]
    assert f"scripts/serving/vllm_sps_run.py --worker --phase {phase}" in body["script"]


def test_stem_selection_is_repeatable_and_covers_subjects():
    rows = [
        {"subject": s, "question": str(i)}
        for s in ("abstract_algebra", "astronomy", "world_religions")
        for i in range(80)
    ]
    indices = select_examples(rows)
    assert indices == select_examples(rows)
    assert len(indices) == len(set(indices)) == c.family_config.EVAL_COUNT
    assert {rows[i]["subject"] for i in indices} == {"abstract_algebra", "astronomy"}


def test_unpinned_eagle_is_rejected():
    exp = next(x for x in c.collect_experiments("eval") if x.arm == "eagle3")
    with pytest.raises(ValueError, match="pinned"):
        engine_arguments(exp, "/hf", None)


@pytest.mark.parametrize("legacy", [False, True])
def test_trained_checkpoint_is_verified_by_its_bytes(tmp_path, monkeypatch, legacy):
    monkeypatch.setattr(c.family_config, "ROOT", tmp_path)
    folder = tmp_path / "runs/out/qwen3_sps_mtp"
    folder.mkdir(parents=True)
    checkpoint = folder / "final.pt"
    checkpoint.write_bytes(b"trained model")
    marker = {
        "checkpoint": str(checkpoint),
        "export": str(folder / "hf"),
        "checkpoint_sha256": common.file_hash(checkpoint),
        "parent": str(c.family_config.PARENT),
        "steps": c.family_config.TRAIN_STEPS,
        "horizons": c.family_config.HORIZONS,
    }
    if legacy:
        marker["design_signature"] = "historical code hash"
    else:
        marker.update(common.training_schedule())
    (folder / "trained.json").write_text(json.dumps(marker))
    assert common.checked_checkpoint(common.trained()) == checkpoint
    with monkeypatch.context() as different_schedule:
        different_schedule.setattr(c.family_config, "GLOBAL_BATCH", 8)
        with pytest.raises(RuntimeError, match="configured run"):
            common.trained()
    checkpoint.write_bytes(b"other model")
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        common.checked_checkpoint(common.trained())


@pytest.mark.parametrize("config, field, changed", [
    (c, "GLOBAL_BATCH", 8),
    (c, "BACKEND", "sdpa"),
    (c.source, "CONTEXT", 4096),
])
def test_legacy_probe_matches_its_measured_workload(tmp_path, monkeypatch, config, field, changed):
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"seconds": [1, 2], "projected_hours": 0}))
    assert common.probe_seconds(probe) == c.family_config.STAGE_STEPS + 2 * (c.family_config.TRAIN_STEPS - c.family_config.STAGE_STEPS)
    with pytest.raises(RuntimeError, match="another workload"):
        common.probe_seconds(probe, num_gpus=1)
    monkeypatch.setattr(config.family_config, field, changed)
    with pytest.raises(RuntimeError, match="another workload"):
        common.probe_seconds(probe)
