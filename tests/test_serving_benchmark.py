"""Serving source/binary identity, oracle isolation and one-GPU routing."""

import base64
import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts import serving_benchmark as bench


def test_all_serving_jobs_are_single_gpu_and_use_separate_child_interpreters():
    for phase in ("audit", "eval"):
        for exp in bench.selected(phase):
            body = bench.payload(exp, Path("/staged"), phase)
            assert body["instance_type"] == "a100.1gpu"
            assert str(bench.launch.PYTHON) in body["script"]
            assert "--worker" in body["script"]
            assert exp.num_gpus == 1
    assert {x.name for x in bench.selected("eval")} == {
        x.name for x in bench.c.collect_experiments("eval")
    }
    assert [x.arm for x in bench.selected("audit")] == ["audit"]


def test_runtime_rejects_mutated_python_or_compiled_extensions(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(bench.runtime, "RUNTIME", root)
    monkeypatch.setattr(bench.runtime, "ARTIFACTS", artifacts)
    site = root / "lib/python3.12/site-packages"
    (root / "bin").mkdir(parents=True)
    (root / "bin/python").touch()
    hashes = {}
    for name in bench.runtime.OVERLAY:
        target = site / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(name)
        hashes[name] = bench.common.file_hash(target)
    extension = site / "vllm/_C_stable_libtorch.abi3.so"
    extension.write_bytes(b"original CUDA binary")
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(extension.read_bytes()).digest())
        .decode()
        .rstrip("=")
    )
    info = site / "vllm-test.dist-info"
    info.mkdir()
    with (info / "RECORD").open("w") as stream:
        csv.writer(stream).writerow(
            ["vllm/_C_stable_libtorch.abi3.so", "sha256=" + digest, "20"]
        )
    (info / "direct_url.json").write_text(json.dumps({"url": bench.runtime.WHEEL}))
    manifest = {
        "design_signature": bench.runtime.signature(),
        "site_packages": str(site),
        "overlay_sha256": hashes,
    }
    bench.launch.store_json(artifacts / "prepared.json", manifest)
    bench.launch.store_json(root / "sps-runtime.json", manifest)
    assert bench.verify_runtime() == manifest
    source = site / bench.runtime.OVERLAY[0]
    original = source.read_text()
    source.write_text("changed")
    with pytest.raises(RuntimeError, match="overlay changed"):
        bench.verify_runtime()
    source.write_text(original)
    extension.write_bytes(b"changed CUDA binary")
    with pytest.raises(RuntimeError, match="extension changed"):
        bench.verify_runtime()


def test_oracle_rejects_mismatched_checkpoints_and_empty_results(tmp_path, monkeypatch):
    path = tmp_path / "reference.json"
    monkeypatch.setattr(bench, "reference_path", lambda: path)
    monkeypatch.setattr(bench.common, "signature", lambda: "training")
    monkeypatch.setattr(bench, "protocol_signature", lambda: "protocol")
    samples = [
        {"prompt_token_ids": [1], "expected_token_ids": [2]}
        for _ in range(bench.c.AUDIT_COUNT)
    ]
    data = {
        "design_signature": "training",
        "checkpoint_sha256": "checkpoint",
        "protocol_signature": "protocol",
        "samples": samples,
    }
    path.write_text(json.dumps(data))
    assert bench.reference_tokens({"checkpoint_sha256": "checkpoint"}) == samples
    with pytest.raises(RuntimeError, match="provenance"):
        bench.reference_tokens({"checkpoint_sha256": "other"})
    data["samples"] = []
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="Incomplete"):
        bench.reference_tokens({"checkpoint_sha256": "checkpoint"})


def test_evaluator_uses_cached_training_oracle_and_restores_binding(monkeypatch):
    from scripts import vllm_sps_evaluate

    original = bench.native_audit.reference
    samples = [{"expected_token_ids": [7]}]
    monkeypatch.setattr(bench, "reference_tokens", lambda trained: samples)
    calls = []

    def evaluate(exp):
        calls.append(exp)
        assert bench.native_audit.reference({"checkpoint": "checked"}) is samples

    monkeypatch.setattr(vllm_sps_evaluate, "evaluate", evaluate)
    exp = bench.selected("eval")[0]
    bench.evaluate(exp)
    assert calls == [exp]
    assert bench.native_audit.reference is original


def test_completed_results_cannot_be_reused_after_protocol_changes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(bench.c, "ROOT", tmp_path)
    monkeypatch.setattr(bench.common, "signature", lambda: "training")
    monkeypatch.setattr(bench, "protocol_signature", lambda: "current")
    exp = bench.selected("eval")[0]
    root = bench.root_for(exp)
    bench.launch.store_json(root / "evaluated.json", {"design_signature": "training"})
    manifest = {"runtime": "tested"}
    bench.launch.store_json(root / "serving-runtime.json", manifest)
    bench.launch.store_json(
        root / "serving-protocol.json", {"protocol_signature": "old"}
    )
    with pytest.raises(RuntimeError, match="another serving protocol"):
        bench.completed(exp, manifest)
    bench.launch.store_json(
        root / "serving-protocol.json", {"protocol_signature": "current"}
    )
    assert bench.completed(exp, manifest)
