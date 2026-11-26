"""Public named-launch paths, including duplicate historic names and archived runs."""

import json
import os
import subprocess
import sys

import pytest

from orchestration.named.registry import named_runs, resolve
from orchestration.suite_registry import WORKER_SCRIPTS


def test_catalog_covers_common_workers_and_disambiguates_archives():
    runs = named_runs()
    assert {family for family, _ in runs.values()} == set(WORKER_SCRIPTS)
    with pytest.raises(ValueError, match="Ambiguous"):
        resolve("smollm2_original_ar", runs)
    family, run = resolve("ar-sps-360m-ce/smollm2_original_ar", runs)
    assert family == "ar-sps-360m-ce"
    assert run.name == "smollm2_original_ar"
    assert "mmlu-full-resume/qwen0.6_ce" not in runs


@pytest.mark.parametrize("entry, gpus", [("scripts/run_eval.py", 1), ("scripts/run_train_jobs.py", 4)])
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
        [sys.executable, "scripts/run_eval.py", "--experiment", name, "--dry"], text=True
    )
    plan = json.loads(out)
    from experiments import instruct_config

    assert plan["settings"]["generation"] == instruct_config.GENERATION["nonthinking"]
    assert plan["settings"]["REVISION"] == instruct_config.REVISION
    assert plan["payload"]["instance_type"] == "a100.1gpu"
    assert (
        "scripts/readout/real_readout_gsm8k.py --worker --filter A-nonthinking"
        in plan["payload"]["script"]
    )


@pytest.mark.parametrize("suite", ["mtp-unified", "spec-baselines"])
def test_inherited_mtp_settings_invalidate_markers(suite):
    code = """
from pathlib import Path
from unittest.mock import patch
from orchestration import experiment_suite as registry
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
    from orchestration.named import cli
    from types import SimpleNamespace
    from orchestration.named import special_families

    monkeypatch.setattr(
        special_families,
        "load",
        lambda _: SimpleNamespace(
            __name__="workloads.distillation.dflash_mask_ablation", experiment_configs=[]
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/run_eval.py", "--experiment", "dflash-mask-ablation/paired-masks"],
    )
    calls = []
    monkeypatch.setattr(cli.subprocess, "call", lambda argv: calls.append(argv) or 0)
    with monkeypatch.context() as context:
        context.setattr(os, "environ", os.environ.copy())
        assert cli.main("eval") == 0
    assert calls == [
        [sys.executable, "scripts/distillation/dflash_mask_ablation.py", "--submit"]
    ]


def test_pilot_signature_tracks_final_evaluation_arguments():
    code = """
from unittest.mock import patch
from orchestration import experiment_suite
original = experiment_suite.eval_command
previous = experiment_suite.design_signature()
with patch.object(
    experiment_suite, "eval_command", lambda run, task: [*original(run, task), "+limit=20"]
):
    assert experiment_suite.design_signature() != previous
"""
    subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, SPS_EXPERIMENT_SUITE="1b"),
        check=True,
    )


def test_cap_receipts_distinguish_paired_cohorts(tmp_path, monkeypatch, capsys):
    from orchestration.named import cap_jobs

    monkeypatch.setattr(cap_jobs.compare, "ROOT", tmp_path)
    prompts = tmp_path / "cohort.json"
    monkeypatch.setattr(cap_jobs.compare, "PROMPTS", prompts)
    monkeypatch.setattr(sys, "argv", ["cap_jobs", "compare", "original", "--dry"])
    descriptions = []
    for cohort in (["same"], ["same"], ["changed"]):
        prompts.write_text(json.dumps(cohort))
        cap_jobs.main()
        payload = json.loads(capsys.readouterr().out)
        assert payload["env_variables"]["HF_HUB_OFFLINE"] == "1"
        assert payload["env_variables"]["VLLM_USE_V2_MODEL_RUNNER"] == "0"
        assert "timeout 4h" in payload["script"]
        descriptions.append(payload["job_desc"])
    assert descriptions[0] == descriptions[1]
    assert descriptions[0] != descriptions[2]


@pytest.mark.parametrize(
    "name,hours",
    [
        ("qwen3-teacher-cap-probe/continuations", 20),
        ("qwen35-unloop-cap-compare/original", 4),
    ],
)
def test_named_cap_preview_matches_worker_timeout(name, hours):
    out = subprocess.check_output(
        [sys.executable, "scripts/run_eval.py", "--experiment", name, "--dry"], text=True
    )
    preview, end = json.JSONDecoder().raw_decode(out)
    payload = json.loads(out[end:])
    assert preview["settings"]["EVAL_HOURS"] == hours
    assert f"timeout {hours}h" in payload["script"]


def test_teacher_cap_receipt_reuses_published_report_identity(monkeypatch, capsys):
    from hashlib import sha256
    from orchestration.named import cap_jobs

    monkeypatch.setattr(sys, "argv", ["cap_jobs", "teacher", "--dry"])
    cap_jobs.main()
    teacher = cap_jobs.teacher
    assert teacher.source_hash(teacher.__file__) == teacher.PILOT_HASH
    path = f"{teacher.PILOT_HASH}/report.json"
    identity = sha256(path.encode()).hexdigest()
    payload = json.loads(capsys.readouterr().out)
    assert identity in payload["job_desc"]
    assert "timeout 20h" in payload["script"]


@pytest.mark.parametrize("arm", [None, "original", "unloop"])
def test_cap_reuses_active_published_receipt(arm, tmp_path, monkeypatch, capsys):
    from types import ModuleType
    from orchestration.named import cap_jobs

    job_name = "lm-mpi-job-22222222-2222-4222-8222-222222222222"
    if arm:
        identity = f"qwen35-cap-{arm}"
        commit = cap_jobs.PUBLISHED_COMPARISON_COMMIT
        marker_dir = tmp_path.joinpath(*cap_jobs.PUBLISHED_COMPARISON_DESIGN, arm)
        monkeypatch.setattr(cap_jobs.compare, "ROOT", tmp_path)
        prompts = tmp_path / "cohort.json"
        prompts.write_text("[]")
        monkeypatch.setattr(cap_jobs.compare, "PROMPTS", prompts)
        monkeypatch.setattr(cap_jobs.compare, "prepare", lambda: None)
        monkeypatch.setattr(cap_jobs.compare, "folder", lambda _: marker_dir)
    else:
        identity = "qwen3-teacher-cap-probe"
        commit = cap_jobs.PUBLISHED_TEACHER_COMMIT
        monkeypatch.setattr(cap_jobs.teacher, "ROOT", tmp_path)
    submissions = tmp_path / "submissions"
    submissions.mkdir()
    (submissions / f"{job_name}.json").write_text(
        json.dumps(
            dict(
                identity=identity,
                commit=commit,
                job_name=job_name,
            )
        )
    )
    checked = []

    class Client:
        def get_job_status(self, name):
            checked.append(name)
            return {"status": "running"}

    utils = ModuleType("mls.manager.job.utils")
    utils.get_in_progress_jobs = lambda **_: []
    utils.training_job_api_from_profile = lambda _: (Client(), {})
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setattr(cap_jobs.launch, "disable_submission_retries", lambda _: None)
    monkeypatch.setattr(
        cap_jobs.launch, "stage_repo", lambda *_: pytest.fail("Staged a duplicate job")
    )
    monkeypatch.setattr(
        sys, "argv", ["cap_jobs", "compare", arm] if arm else ["cap_jobs", "teacher"]
    )

    cap_jobs.main()

    assert checked == [job_name]
    assert capsys.readouterr().out.strip() == job_name


def test_cap_report_identities_track_inherited_settings(tmp_path, monkeypatch):
    from pathlib import Path
    from orchestration.named import cap_jobs
    from orchestration.source_compatibility import source_hash

    teacher, compare = cap_jobs.teacher, cap_jobs.compare
    monkeypatch.setattr(compare, "ROOT", tmp_path)
    prompts = tmp_path / "cohort.json"
    prompts.write_text("[]")
    monkeypatch.setattr(compare, "PROMPTS", prompts)
    teacher_identity = source_hash(teacher.__file__)
    paired_folder = compare.folder("original")
    original = Path.read_bytes

    def change_sampler(path):
        data = original(path)
        if path == Path(teacher.c.__file__):
            return data + b"\nTEACHER_TOP_P = 0.75\n"
        return data

    monkeypatch.setattr(Path, "read_bytes", change_sampler)
    assert source_hash(teacher.__file__) != teacher_identity
    assert compare.folder("original") != paired_folder


def test_cap_completed_comparison_skips_cohort_preparation(
    tmp_path, monkeypatch, capsys
):
    from orchestration.named import cap_jobs

    compare = cap_jobs.compare
    monkeypatch.setattr(compare, "ROOT", tmp_path)
    prompts = tmp_path / "cohort.json"
    prompts.write_text("[]")
    monkeypatch.setattr(compare, "PROMPTS", prompts)
    folder = compare.folder("original")
    folder.mkdir(parents=True)
    report = folder / "report.json"
    report.write_text("{}")
    monkeypatch.setattr(
        compare, "prepare", lambda: pytest.fail("Reconstructed the cohort")
    )
    monkeypatch.setattr(sys, "argv", ["cap_jobs", "compare", "original"])
    cap_jobs.main()
    assert capsys.readouterr().out.strip() == f"Completed original: {report}"


def test_teacher_cap_assists_require_archived_design(monkeypatch):
    from orchestration.named import cap_jobs

    teacher = cap_jobs.teacher
    monkeypatch.setattr(teacher, "source_hash", lambda _: "different-design")
    with pytest.raises(RuntimeError, match="published teacher-cap design"):
        teacher.run(shard=0)


def test_unprepared_drafter_prints_worker_command(tmp_path):
    from types import SimpleNamespace
    from orchestration.named.distillation import preview_drafter
    from orchestration.named.standalone import Job

    module = SimpleNamespace(
        __name__="workloads.distillation.qwen4b_drafter",
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
        "scripts/distillation/qwen4b_drafter.py",
        *plan.args,
        "--worker",
    ]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode", ["thinking", "nonthinking"])
def test_provenance_preview_uses_evaluation_generation(mode):
    from orchestration.named.special_families import provenance
    from workloads.provenance import provenance_evaluate, provenance_prepare

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
    from workloads.serving import vllm_sps_train

    out = subprocess.check_output(
        [
            sys.executable,
            "scripts/run_train_jobs.py",
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
