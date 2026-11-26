"""Public named-launch paths, including duplicate historic names and archived runs."""

import json
import os
import subprocess
import sys

import pytest

from experiment_defs.registry import named_runs, resolve
from scripts.suite_registry import WORKER_MODULES


def test_catalog_covers_common_workers_and_disambiguates_archives():
    runs = named_runs()
    assert {family for family, _ in runs.values()} == set(WORKER_MODULES)
    with pytest.raises(ValueError, match="Ambiguous"):
        resolve("smollm2_original_ar", runs)
    family, run = resolve("ar-sps-360m-ce/smollm2_original_ar", runs)
    assert family == "ar-sps-360m-ce"
    assert run.name == "smollm2_original_ar"
    assert "mmlu-full-resume/qwen0.6_ce" not in runs


@pytest.mark.parametrize("entry, gpus", [("run_eval.py", 1), ("run_train_jobs.py", 4)])
def test_named_preview_overrides_ambient_suite(entry, gpus):
    env = dict(os.environ, SPS_EXPERIMENT_SUITE="10b")
    name = "s_standard_pilot_1b"
    out = subprocess.check_output(
        [sys.executable, entry, "--experiment", name, "--dry"], env=env, text=True
    )
    plan, _ = json.JSONDecoder().raw_decode(out)
    assert plan["payload"]["instance_type"] == f"a100.{gpus}gpu"
    assert plan["payload"]["env_variables"]["SPS_EXPERIMENT_SUITE"] == "1b"
    assert plan["settings"][name]["training"]["max_tokens"] == 1_000_341_504
    assert plan["settings"][name]["model"]["config"]["n_layer"] == 12
    if gpus == 1:
        commands = plan["settings"][name]["worker_commands"]
        assert all(
            "+checkpoint=final" in cmd and "+num_fewshot=0" in cmd for cmd in commands
        )
        assert any("+tasks=piqa" in cmd for cmd in commands)


def test_readout_named_preview_uses_its_checkpoint_and_worker():
    name = "qwen3-instruct-real-readout-mtp/gsm8k-A-nonthinking"
    out = subprocess.check_output(
        [sys.executable, "run_eval.py", "--experiment", name, "--dry"], text=True
    )
    plan = json.loads(out)
    assert plan["payload"]["instance_type"] == "a100.1gpu"
    assert (
        "scripts.real_readout_gsm8k --worker --filter A-nonthinking"
        in plan["payload"]["script"]
    )


@pytest.mark.parametrize("suite", ["mtp-unified", "spec-baselines"])
def test_inherited_mtp_settings_invalidate_markers(suite):
    code = """
from pathlib import Path
from unittest.mock import patch
import experiments as registry
original = Path.read_bytes
previous = registry.design_signature()
def changed(path):
    data = original(path)
    if path.parts[-2:] == ('experiment_defs', 'mtp.py'):
        return data + b'changed source'
    return data
with patch.object(Path, 'read_bytes', changed):
    assert registry.design_signature() != previous
"""
    subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, SPS_EXPERIMENT_SUITE=suite),
        check=True,
    )


def test_named_mask_ablation_selects_submission_mode(monkeypatch):
    from experiment_defs import cli
    from types import SimpleNamespace
    from experiment_defs import special_families

    monkeypatch.setattr(
        special_families,
        "load",
        lambda _: SimpleNamespace(
            __name__="scripts.distillation.dflash_mask_ablation", experiment_configs=[]
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_eval.py", "--experiment", "dflash-mask-ablation/paired-masks"],
    )
    calls = []
    monkeypatch.setattr(cli.subprocess, "call", lambda argv: calls.append(argv) or 0)
    with monkeypatch.context() as context:
        context.setattr(os, "environ", os.environ.copy())
        assert cli.main("eval") == 0
    assert calls == [
        [sys.executable, "-m", "scripts.distillation.dflash_mask_ablation", "--submit"]
    ]
