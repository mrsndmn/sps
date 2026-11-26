"""Isolation and model identity in the four-arm vLLM job design."""

import json
from pathlib import Path

import pytest

from scripts import vllm_sps_config as c
from scripts.serving import vllm_sps_common as common
from scripts.vllm_sps_evaluate import engine_arguments, select_examples
from scripts.vllm_sps_run import payload


def test_eval_jobs_are_single_gpu_and_ordinary_sps_shares_speculative_checkpoint():
    exports = "/immutable/final/hf"
    for exp in c.collect_experiments("eval"):
        body = payload("eval", exp, Path("/staged"), "author", "nick")
        assert body["instance_type"] == "a100.1gpu"
        assert body["env_variables"]["VLLM_USE_V2_MODEL_RUNNER"] == "0"
        args = engine_arguments(exp, exports, "eagle-immutable-revision")
        assert args["max_num_seqs"] == args["tensor_parallel_size"] == 1
        assert not args["enable_prefix_caching"]
        assert not args["async_scheduling"]
        assert args["model"] == (exports if exp.arm.startswith("sps") else c.MODEL)
        if exp.arm == "sps":
            assert "speculative_config" not in args
        if exp.arm == "eagle3":
            assert args["speculative_config"]["revision"] == "eagle-immutable-revision"


def test_stem_selection_is_repeatable_and_covers_subjects():
    rows = [
        {"subject": s, "question": str(i)}
        for s in ("abstract_algebra", "astronomy", "world_religions")
        for i in range(80)
    ]
    indices = select_examples(rows)
    assert indices == select_examples(rows)
    assert len(indices) == len(set(indices)) == c.EVAL_COUNT
    assert {rows[i]["subject"] for i in indices} == {"abstract_algebra", "astronomy"}


def test_unpinned_eagle_is_rejected():
    exp = next(x for x in c.collect_experiments("eval") if x.arm == "eagle3")
    with pytest.raises(ValueError, match="pinned"):
        engine_arguments(exp, "/hf", None)


@pytest.mark.parametrize("legacy", [False, True])
def test_trained_checkpoint_is_verified_by_its_bytes(tmp_path, monkeypatch, legacy):
    monkeypatch.setattr(c, "ROOT", tmp_path)
    folder = tmp_path / "runs/out/qwen3_sps_mtp"
    folder.mkdir(parents=True)
    checkpoint = folder / "final.pt"
    checkpoint.write_bytes(b"trained model")
    marker = {
        "checkpoint": str(checkpoint),
        "export": str(folder / "hf"),
        "checkpoint_sha256": common.file_hash(checkpoint),
        "parent": str(c.PARENT),
        "steps": c.TRAIN_STEPS,
        "horizons": c.HORIZONS,
    }
    if legacy:
        marker["design_signature"] = "historical code hash"
    (folder / "trained.json").write_text(json.dumps(marker))
    assert common.checked_checkpoint(common.trained()) == checkpoint
    checkpoint.write_bytes(b"other model")
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        common.checked_checkpoint(common.trained())
