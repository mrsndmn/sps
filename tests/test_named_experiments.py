"""Public named-launch paths, including duplicate historic names and archived runs."""

import json
import os
import subprocess
import sys

import pytest

from scripts.named.registry import named_runs, resolve
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
    from scripts import instruct_config

    assert plan["settings"]["generation"] == instruct_config.GENERATION["nonthinking"]
    assert plan["settings"]["REVISION"] == instruct_config.REVISION
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
from scripts import experiment_suite as registry
original = Path.read_bytes
previous = registry.design_signature()
def changed(path):
    data = original(path)
    if path.parts[-2:] == ('experiments', 'mtp_design.py'):
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
    from scripts.named import cli
    from types import SimpleNamespace
    from scripts.named import special_families

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


def test_pilot_signature_tracks_final_evaluation_arguments(monkeypatch):
    from scripts import experiment_suite as experiments

    original = experiments.eval_command
    previous = experiments.design_signature()
    monkeypatch.setattr(
        experiments,
        "eval_command",
        lambda run, task: [*original(run, task), "+limit=20"],
    )
    assert experiments.design_signature() != previous


def test_cap_receipts_distinguish_paired_cohorts(tmp_path, monkeypatch, capsys):
    from scripts.named import cap_jobs

    monkeypatch.setattr(cap_jobs.compare, "ROOT", tmp_path)
    prompts = tmp_path / "cohort.json"
    monkeypatch.setattr(cap_jobs.compare, "PROMPTS", prompts)
    monkeypatch.setattr(sys, "argv", ["cap_jobs", "compare", "original", "--dry"])
    descriptions = []
    for cohort in (["same"], ["same"], ["changed"]):
        prompts.write_text(json.dumps(cohort))
        cap_jobs.main()
        descriptions.append(json.loads(capsys.readouterr().out)["job_desc"])
    assert descriptions[0] == descriptions[1]
    assert descriptions[0] != descriptions[2]


def test_teacher_cap_receipt_uses_worker_report_identity(monkeypatch, capsys):
    from hashlib import sha256
    from scripts.named import cap_jobs

    monkeypatch.setattr(sys, "argv", ["cap_jobs", "teacher", "--dry"])
    cap_jobs.main()
    path = f"{cap_jobs.teacher.source_hash(cap_jobs.teacher.__file__)}/report.json"
    identity = sha256(path.encode()).hexdigest()
    assert identity in json.loads(capsys.readouterr().out)["job_desc"]


def test_unprepared_drafter_prints_worker_command(tmp_path):
    from types import SimpleNamespace
    from scripts.named.distillation import preview_drafter
    from scripts.named.standalone import Job

    module = SimpleNamespace(
        __name__="scripts.distillation.qwen4b_drafter",
        training=SimpleNamespace(DATA=tmp_path, DEFAULT_DATASET="smoltalk"),
    )
    plan = Job(
        "smoltalk",
        "eval",
        module.__name__,
        ("--phase", "eval", "--dataset", "smoltalk"),
        {},
    )
    preview = preview_drafter(
        plan, module, tmp_path, SimpleNamespace(PYTHON=sys.executable)
    )
    assert preview["pending_input"] == str(tmp_path / "manifest.json")
    assert preview["worker_command"] == [
        sys.executable,
        "-m",
        module.__name__,
        *plan.args,
        "--worker",
    ]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode", ["thinking", "nonthinking"])
def test_provenance_preview_uses_evaluation_generation(mode):
    from scripts.named.special_families import provenance
    from scripts import provenance_evaluate, provenance_prepare

    plans = list(provenance())
    prep = next(plan for plan in plans if plan.phase == "prepare")
    assert prep.settings["GENERATION"] == provenance_prepare.GENERATION
    plan = next(plan for plan in plans if plan.name == "continued_ar-teacher--" + mode)
    assert plan.settings["generation"] == provenance_evaluate.c.GENERATION[mode]
    assert plan.settings["preparation_generation"] == provenance_prepare.GENERATION
    assert plan.settings["GENERATION"] == provenance_evaluate.c.GENERATION


@pytest.mark.parametrize(
    "family,run",
    [
        ("vllm-sps-qwen3", "qwen3_sps_mtp"),
        ("vllm-sps-draft-sweep", "qwen3_sps_mtp"),
        ("qwen3-instruct-real-readout-mtp", "qwen3_real_readout_mtp"),
    ],
)
@pytest.mark.parametrize("phase", ["train", "probe"])
def test_named_train_preview_includes_shared_optimizer_and_context(family, run, phase):
    from scripts import vllm_sps_train

    out = subprocess.check_output(
        [
            sys.executable,
            "run_train_jobs.py",
            "--experiment",
            family + "/" + run,
            "--phase",
            phase,
            "--dry",
        ],
        text=True,
    )
    plan, _ = json.JSONDecoder().raw_decode(out)
    actual = plan["settings"]
    assert "GPU_HOURS" not in actual and "EVAL_MAX_HOURS" not in actual
    for field in ("CONTEXT", "EMBED_LR", "FULL_LR", "BETAS", "WEIGHT_DECAY", "CLIP"):
        expected = getattr(vllm_sps_train.source, field)
        assert actual[field] == (
            list(expected) if isinstance(expected, tuple) else expected
        )
