"""Separate graph experiment identity and actual replay evidence gates."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import serving_graphs as graphs


def test_graph_arguments_preserve_sampling_target_and_enable_full_decode_capture():
    for exp in graphs.bench.selected("eval"):
        result = graphs.graph_arguments(
            graphs.engine_arguments, exp, "/export", "revision"
        )
        original = graphs.engine_arguments(exp, "/export", "revision")
        assert result.pop("enforce_eager") is False
        assert result.pop("compilation_config") == graphs.COMPILATION
        original.pop("enforce_eager")
        assert result == original
    assert graphs.ROOT != graphs.EAGER_ROOT


def test_graph_completion_requires_current_protocol_and_real_target_replays(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(graphs, "protocol_signature", lambda: "current")
    path = tmp_path / "cudagraph.json"
    value = {
        "protocol_signature": "current",
        "workers": [{"captured": 1, "target_replays": 0, "mode": "FULL_DECODE_ONLY"}],
    }
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="actual target CUDA graph replay"):
        graphs.validate_evidence(path)
    value["workers"][0]["target_replays"] = 3
    path.write_text(json.dumps(value))
    graphs.validate_evidence(path)
    value["protocol_signature"] = "old"
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="another protocol"):
        graphs.validate_evidence(path)


def test_child_uses_graph_entrypoint_and_source_without_changing_parent(monkeypatch):
    calls = []
    monkeypatch.setattr(graphs, "BASE_CHILD", lambda *args: calls.append(args))
    env = {"PYTHONPATH": "/staged:/staged/src"}
    graphs.child(
        ["/python", "-m", "scripts.serving_benchmark", "--eval-worker"], env, 1
    )
    command, settings, hours = calls[0]
    assert command[2] == "scripts.serving_graphs"
    assert settings["PYTHONPATH"].split(os.pathsep) == [
        str(graphs.SITE),
        "/staged",
        "/staged/src",
    ]
    assert env == {"PYTHONPATH": "/staged:/staged/src"}
    assert hours == 1


def test_warmup_counts_only_target_replays_and_restores_torch_method(monkeypatch):
    class Graph:
        def replay(self):
            return "replayed"

    class Wrapper:
        def __init__(self, target):
            self.concrete_cudagraph_entries = {
                "decode": SimpleNamespace(cudagraph=target)
            }

    original = Graph.replay
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(CUDAGraph=Graph))
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.compilation.cuda_graph",
        SimpleNamespace(CUDAGraphWrapper=Wrapper),
    )
    target, draft = Graph(), Graph()
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            model=Wrapper(target),
            compilation_config=SimpleNamespace(cudagraph_mode="FULL_DECODE_ONLY"),
        )
    )
    graphs.start_tracking(worker)
    try:
        assert target.replay() == draft.replay() == target.replay() == "replayed"
    finally:
        state = graphs.stop_tracking(worker)
    assert state["captured"] == 1
    assert state["target_replays"] == 2
    assert Graph.replay is original


def test_graph_cli_dry_payload_is_one_gpu_and_distinct_from_eager():
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    for phase in ("audit", "eval"):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.serving_graphs", "--phase", phase, "--dry"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        assert "sps-serving-benchmark graphs " in result.stdout
        assert "scripts.serving_graphs" in result.stdout
        assert '"instance_type": "a100.1gpu"' in result.stdout
        assert "vllm-sps-qwen3-cudagraphs" not in str(graphs.EAGER_ROOT)
