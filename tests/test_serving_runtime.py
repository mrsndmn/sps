"""Runtime preparation isolation and binary/source provenance gates."""

import json
import os
from pathlib import Path

import pytest

from scripts import serving_runtime as runtime


def test_runtime_job_uses_one_gpu_and_preserves_project_interpreter():
    payload = runtime.payload(Path("/staged"))
    assert payload["instance_type"] == "a100.1gpu"
    assert str(runtime.launch.PYTHON) in payload["script"]
    assert "--worker" in payload["script"]
    assert runtime.RUNTIME.parent == runtime.ENV_ROOT
    assert runtime.c.VLLM_BASE in runtime.WHEEL


def test_runtime_rejects_overwriting_existing_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "RUNTIME", tmp_path)
    (tmp_path / "pyvenv.cfg").write_text("home = /other/python")
    with pytest.raises(RuntimeError, match="unowned"):
        runtime.worker()
    (tmp_path / ".sps-runtime-owner.json").write_text(json.dumps({"other": "owner"}))
    with pytest.raises(RuntimeError, match="unrelated"):
        runtime.worker()


def test_precompiled_runtime_rejects_unaccounted_source_or_binary_changes(monkeypatch):
    changed = list(runtime.OVERLAY) + ["docs/features/speculative_decoding/sps.md"]
    monkeypatch.setattr(runtime.launch, "git", lambda *args: "\n".join(changed))
    runtime.validate_overlay()
    changed.append("csrc/attention.cu")
    with pytest.raises(RuntimeError, match="Unexpected compiled/runtime"):
        runtime.validate_overlay()


def test_overlay_does_not_mutate_cached_wheel_or_other_environments(tmp_path):
    cached = tmp_path / "cached.py"
    cached.write_text("upstream")
    installed = tmp_path / "installed.py"
    other = tmp_path / "other.py"
    os.link(cached, installed)
    os.link(cached, other)
    source = tmp_path / "source.py"
    source.write_text("SPS implementation")
    runtime.replace_overlay(source, installed)
    assert installed.read_text() == "SPS implementation"
    assert cached.read_text() == other.read_text() == "upstream"
    assert installed.stat().st_ino != cached.stat().st_ino
