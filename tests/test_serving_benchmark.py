"""Serving source/binary identity, oracle isolation and one-GPU routing."""

import base64
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from workloads.serving import serving_benchmark as bench


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
        **bench.runtime.identity(),
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
    monkeypatch.setattr(bench, "reference_path", lambda name: path)
    monkeypatch.setattr(bench, "protocol_signature", lambda: "protocol")
    samples = [
        {"prompt_token_ids": [1], "expected_token_ids": [2]}
        for _ in range(bench.c.family_config.AUDIT_COUNT)
    ]
    data = {
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
    from workloads.serving import vllm_sps_evaluate

    AutoTokenizer = SimpleNamespace()
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoTokenizer=AutoTokenizer)
    )
    original = bench.native_audit.reference
    manifest = {"runtime": "verified"}
    monkeypatch.setattr(bench, "verify_runtime", lambda: manifest)
    monkeypatch.setattr(
        bench,
        "check_gate",
        lambda actual: actual == manifest or pytest.fail("Unverified runtime"),
    )
    samples = [{"expected_token_ids": [7]}]
    monkeypatch.setattr(bench, "reference_tokens", lambda trained: samples)
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, *, return_dict=True, **kwargs):
            return {"input_ids": [42]} if return_dict else [42]

    loader_calls = []

    def load(*args, **kwargs):
        loader_calls.append((args, kwargs))
        return Tokenizer()

    AutoTokenizer.from_pretrained = load

    def evaluate(exp):
        calls.append(exp)
        assert bench.native_audit.reference({"checkpoint": "checked"}) is samples
        tokenizer = AutoTokenizer.from_pretrained("pinned-model", revision="commit")
        assert tokenizer.apply_chat_template([], tokenize=True) == [42]

    monkeypatch.setattr(vllm_sps_evaluate, "evaluate", evaluate)
    exp = bench.selected("eval")[0]
    bench.evaluate(exp)
    assert calls == [exp]
    assert bench.native_audit.reference is original
    assert AutoTokenizer.from_pretrained is load
    assert loader_calls == [(("pinned-model",), {"revision": "commit"})]


def test_completed_results_cannot_be_reused_after_protocol_changes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(bench.c.family_config, "ROOT", tmp_path)
    monkeypatch.setattr(bench, "protocol_signature", lambda: "current")
    monkeypatch.setattr(bench.evaluator, "protocol_signature", lambda: "current")
    monkeypatch.setattr(bench, "trained", lambda: {"checkpoint_sha256": "current"})
    gates = []
    monkeypatch.setattr(bench, "check_gate", lambda manifest, **kwargs: gates.append(manifest))
    exp = bench.selected("eval")[0]
    root = bench.root_for(exp)
    bench.launch.store_json(root / "evaluated.json", {
        "experiment": vars(exp), "checkpoint_sha256": "current", "evaluation_protocol": "current",
    })
    manifest = {"runtime": "tested"}
    bench.launch.store_json(root / "manifest.json", {
        "experiment": vars(exp), "checkpoint_sha256": "current", "evaluation_protocol": "current",
    })
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
    assert gates == [manifest]
    for name in ("evaluated.json", "manifest.json"):
        bench.launch.store_json(root / name, {
            "experiment": vars(exp), "checkpoint_sha256": "current", "evaluation_protocol": "archived",
        })
    with pytest.raises(RuntimeError, match="another evaluator"):
        bench.completed(exp, manifest)
    assert bench.completed(exp, manifest, protocol="current")
    bench.launch.store_json(root / "manifest.json", {
        "experiment": vars(exp), "checkpoint_sha256": "old", "evaluation_protocol": "archived",
    })
    with pytest.raises(RuntimeError, match="another checkpoint"):
        bench.completed(exp, manifest)


def test_serving_protocol_tracks_generation_configuration(monkeypatch):
    from workloads.serving import draft_sweep, serving_graphs

    protocols = (bench.protocol_signature, serving_graphs.protocol_signature,
                 draft_sweep.protocol_signature)
    before = [protocol() for protocol in protocols]
    monkeypatch.setitem(bench.evaluator.c.family_config.GENERATION, "new-mode", {"temperature": 0.5})
    assert all(protocol() != old for protocol, old in zip(protocols, before, strict=True))


def test_audit_replicate_children_and_evidence_use_own_directory(monkeypatch, tmp_path):
    import sys
    from dataclasses import replace
    from experiments import serving_benchmark as config
    from orchestration.named.registry import family_runs
    from workloads.serving import serving_graphs as graphs

    original = bench.selected("audit")[0]
    second = replace(original, name=original.name + "-second")
    rows = [original, second]
    for module in (config, bench):
        monkeypatch.setattr(module, "experiment_configs", rows)
    assert (second.name, "audit") not in family_runs("vllm-sps-draft-sweep")
    monkeypatch.setattr(bench.c.family_config, "ROOT", tmp_path)
    monkeypatch.setattr(bench, "verify_runtime", lambda: {"site_packages": "site"})
    monkeypatch.setattr(bench, "trained", lambda: {"checkpoint_sha256": "trained"})
    monkeypatch.setattr(bench.runtime, "runtime_environment", dict)
    children = []
    monkeypatch.setattr(bench, "child", lambda argv, env, hours: children.append(argv[1:]))
    bench.worker(second)
    assert len(children) == 3
    assert bench.root_for(second).is_dir() and not bench.root_for(original).exists()
    seen = []
    monkeypatch.setattr(bench, "make_reference", lambda name: seen.append(name))
    monkeypatch.setattr(bench, "audit_arm", lambda arm, name: seen.append(name))
    for argv in children:
        monkeypatch.setattr(sys, "argv", argv)
        bench.main()
    assert seen == [second.name] * 3
    assert bench.reference_path(second.name).parent == bench.root_for(second)
    monkeypatch.setattr(graphs, "ROOT", tmp_path)
    evidence = []
    monkeypatch.setattr(graphs, "with_graph_engine", lambda fn, arm, path: evidence.append(path))
    graphs.audit_arm("sps", second.name)
    assert evidence == [bench.root_for(second) / "graphs-sps.json"]
