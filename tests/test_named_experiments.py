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
