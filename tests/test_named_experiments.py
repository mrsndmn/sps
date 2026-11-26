"""Public named-launch paths, including duplicate historic names and archived runs."""

import json
import os
import subprocess
import sys

import pytest

from orchestration.named.registry import FAMILIES, family_runs
from orchestration.suite_registry import WORKER_SCRIPTS


def test_shared_preparation_has_one_name_and_one_output():
    runs = family_runs("1b")
    assert [name for name, phase in runs if phase == "prepare"] == ["data-and-probes"]
    out = subprocess.check_output(
        [
            sys.executable,
            "scripts/run_train_jobs.py",
            "--experiment",
            "1b/data-and-probes",
            "--phase",
            "prepare",
            "--dry",
        ],
        text=True,
    )
    preview = json.loads(out)
    assert preview["output_dir"] and "data-and-probes" in preview["payload"]["job_desc"]


def test_catalog_covers_common_workers_and_disambiguates_archives():
    assert set(WORKER_SCRIPTS) <= FAMILIES.keys()
    assert ("smollm2_original_ar", "eval") in family_runs("ar-sps-360m-ce")
    assert ("qwen0.6_ce", "eval") not in family_runs("mmlu-full-resume")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_eval.py",
            "--experiment",
            "smollm2_original_ar",
            "--dry",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "FAMILY/NAME" in result.stderr


@pytest.mark.parametrize(
    "entry, gpus", [("scripts/run_eval.py", 1), ("scripts/run_train_jobs.py", 4)]
)
def test_named_preview_overrides_ambient_suite(entry, gpus):
    env = dict(os.environ, SPS_EXPERIMENT_SUITE="10b")
    name = "s_standard_pilot_1b"
    out = subprocess.check_output(
        [sys.executable, entry, "--experiment", "1b/" + name, "--dry"],
        env=env,
        text=True,
    )
    plan, _ = json.JSONDecoder().raw_decode(out)
    assert plan["payload"]["instance_type"] == f"a100.{gpus}gpu"
    assert plan["payload"]["env_variables"]["SPS_EXPERIMENT_SUITE"] == "1b"
    assert plan["settings"][name]["training"]["max_tokens"] == 1_000_341_504
    assert plan["settings"][name]["model"]["config"]["n_layer"] == 12
    if gpus == 1:
        commands = plan["settings"][name]["worker_commands"]
        assert all(
            ("--checkpoint_path" in cmd or "--checkpoint" in cmd)
            and cmd[cmd.index("--num_fewshot") + 1] == "0"
            for cmd in commands
        )
        assert any(cmd[cmd.index("--tasks") + 1] == "piqa" for cmd in commands)


def test_readout_named_preview_uses_its_checkpoint_and_worker():
    name = "qwen3-instruct-real-readout-mtp/gsm8k-A-nonthinking"
    out = subprocess.check_output(
        [sys.executable, "scripts/run_eval.py", "--experiment", name, "--dry"],
        text=True,
    )
    plan = json.loads(out)
    from experiments import instruct_config

    assert (
        plan["settings"]["instruct_config"]["generation"]["nonthinking"]
        == instruct_config.family_config.GENERATION["nonthinking"]
    )
    assert (
        plan["settings"]["instruct_config"]["revision"]
        == instruct_config.family_config.REVISION
    )
    assert plan["payload"]["instance_type"] == "a100.1gpu"
    assert (
        "scripts/readout/real_readout_gsm8k.py --worker --filter A-nonthinking"
        in plan["payload"]["script"]
    )


def test_named_mask_ablation_selects_submission_mode(monkeypatch, tmp_path):
    from orchestration.named import cli

    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/run_eval.py", "--experiment", "dflash-mask-ablation/paired-masks"],
    )
    calls = []
    monkeypatch.setattr(
        cli.subprocess, "call", lambda argv, env: calls.append(argv) or 0
    )
    with monkeypatch.context() as context:
        context.setattr(os, "environ", os.environ.copy())
        assert cli.main("eval") == 0
    assert calls == [
        [sys.executable, "scripts/distillation/dflash_mask_ablation.py", "--submit"]
    ]


def test_cap_submission_identity_uses_configured_output(tmp_path, monkeypatch, capsys):
    from orchestration.named import cap_jobs

    monkeypatch.setattr(cap_jobs.compare, "ROOT", tmp_path)
    prompts = tmp_path / "cohort.json"
    monkeypatch.setattr(cap_jobs.compare, "PROMPTS", prompts)
    monkeypatch.setattr(sys, "argv", ["cap_jobs", "compare", "original", "--dry"])
    descriptions = []
    for cohort in (["same"], ["same"], ["changed"]):
        prompts.write_text(json.dumps(cohort))
        cap_jobs.main()
        payload = json.loads(capsys.readouterr().out)["payload"]
        assert payload["env_variables"]["HF_HUB_OFFLINE"] == "1"
        assert payload["env_variables"]["VLLM_USE_V2_MODEL_RUNNER"] == "0"
        assert "timeout 4h" in payload["script"]
        descriptions.append(payload["job_desc"])
    assert descriptions[0] == descriptions[1]
    assert descriptions[0] == descriptions[2]


@pytest.mark.parametrize(
    "name,hours",
    [
        ("qwen3-teacher-cap-probe/continuations", 20),
        ("qwen35-unloop-cap-compare/original", 4),
    ],
)
def test_named_cap_preview_matches_worker_timeout(name, hours):
    out = subprocess.check_output(
        [sys.executable, "scripts/run_eval.py", "--experiment", name, "--dry"],
        text=True,
    )
    preview = json.loads(out)
    payload = preview["payload"]
    group = "teacher_cap_probe" if hours == 20 else "qwen35_cap_compare"
    assert preview["settings"][group]["eval_hours"] == hours
    assert f"timeout {hours}h" in payload["script"]


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


@pytest.mark.parametrize("mode", ["thinking", "nonthinking"])
def test_provenance_preview_uses_evaluation_generation(mode):
    from experiments import instruct_config, provenance_prepare

    name = "instruct-response-provenance/continued_ar-teacher--" + mode
    plan = json.loads(
        subprocess.check_output(
            [sys.executable, "scripts/run_eval.py", "--experiment", name, "--dry"]
        )
    )
    assert (
        plan["settings"]["instruct_config"]["generation"][mode]
        == instruct_config.family_config.GENERATION[mode]
    )
    assert (
        plan["settings"]["provenance_prepare"]["generation"]
        == provenance_prepare.family_config.GENERATION
    )


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
    actual = plan["settings"]["instruct_config"]
    for field in ("CONTEXT", "EMBED_LR", "FULL_LR", "BETAS", "WEIGHT_DECAY", "CLIP"):
        expected = getattr(vllm_sps_train.source.family_config, field)
        assert actual[field.lower()] == (
            list(expected) if isinstance(expected, tuple) else expected
        )


def test_second_dflash_config_reaches_worker_and_owns_its_output(tmp_path):
    """A new row in one family file is sufficient for list, launch and dedup."""
    import shlex
    import shutil
    from pathlib import Path

    from orchestration.run_directory import RunDirectory

    repo = Path(__file__).resolve().parents[1]
    shutil.copytree(
        repo / "src/experiments",
        tmp_path / "src/experiments",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    launcher = "scripts/run_train_jobs.py"
    native = "scripts/distillation/dflash_repro.py"
    for script in (launcher, native):
        (tmp_path / script).parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(repo / script, tmp_path / script)
    artifacts = tmp_path / "artifacts"
    config = tmp_path / "src/experiments/dflash_repro_train.py"
    with config.open("a") as stream:
        stream.write(
            '\nexperiment_configs.append(Experiment(name="second", '
            f"output_root=Path({str(artifacts / 'custom')!r}), "
            'eval_dir_pattern="eval-{step}", eval_protocol=GSM8KProtocol(limit=17), lr=0.0123, num_gpus=1, milestones=[]))\n'
        )
    # Isolate NFS state while keeping both entry points and their real parsers.
    (tmp_path / "sitecustomize.py").write_text(
        f"from experiments import environment\n"
        f"environment.COMMON_ROOT = environment.Path({str(artifacts)!r})\n"
    )
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            map(str, (tmp_path, tmp_path / "src", repo / "src"))
        ),
        GIT_DIR=subprocess.check_output(
            ["git", "rev-parse", "--absolute-git-dir"], cwd=repo, text=True
        ).strip(),
        SPS_EXPERIMENT_SUITE="ar-sps-qwen3-1.7b-instruct",
    )

    def call(*args):
        return subprocess.check_output(
            [sys.executable, *args], cwd=tmp_path, env=env, text=True
        )

    family = "dflash-reproduction"
    name = family + "/second"
    listing = call(
        "-c",
        "import runpy, sys; sys.modules.update(dict.fromkeys(['torch', 'mls', 'transformers'])); "
        "sys.argv.pop(0); runpy.run_path(sys.argv[0], run_name='__main__')",
        launcher,
        "--list",
        family,
    )
    assert {
        line.split()[1] for line in listing.splitlines() if line.startswith(name + "\t")
    } == {"probe", "train", "eval"}
    plan = json.loads(call(launcher, "--experiment", name, "--dry"))
    assert plan == json.loads(
        call(native, "--phase", "train", "--filter", "second", "--dry")
    )
    output = artifacts / "custom/runs/out/second"
    assert "--cfg." not in plan["payload"]["script"]
    assert plan["settings"]["lr"] == 0.0123
    assert plan["output_dir"] == str(output)
    assert plan["payload"]["instance_type"] == "a100.1gpu"
    command = shlex.split(shlex.split(plan["payload"]["script"])[-1])
    argv = command[command.index(native) + 1 :]
    worker = """
import json, sys
from workloads.distillation import dflash_repro as jobs
jobs.training.train = lambda experiment, probe: print(json.dumps([experiment.name, experiment.lr, experiment.milestones, str(experiment.output("train")), probe]))
jobs.main()
"""
    assert json.loads(call("-c", worker, *argv)) == [
        "second",
        0.0123,
        [],
        str(output),
        False,
    ]
    directory = RunDirectory(artifacts, output)
    directory.reservation.mkdir(parents=True)
    assert "Skip unresolved reservation" in call(launcher, "--experiment", name)
    evaluate = (launcher, "--experiment", name, "--phase", "eval", "--dry")
    assert json.loads(call(*evaluate)) == {
        "pending_input": str(output / "trained.json")
    }
    output.mkdir(parents=True)
    (output / "trained.json").write_text('{"steps": 42}')
    eval_plan = json.loads(call(*evaluate))
    eval_output = output.parent / "eval-42"
    assert eval_plan["output_dir"] == str(eval_output)
    assert "lr" not in eval_plan["settings"]
    assert eval_plan["settings"]["limit"] == 17
    command = shlex.split(shlex.split(eval_plan["payload"]["script"])[-1])
    assert json.loads(call(
        "-c",
        "import json; from workloads.distillation import dflash_repro as jobs; "
        "jobs.evaluation.evaluate = lambda args: print(json.dumps([args.name, str(args.checkpoint), str(args.result)])); jobs.main()",
        *command[command.index(native) + 1 :],
    )) == ["second", str(output / "step-00042"), str(eval_output / "evaluated.json")]
    assert "Skip existing output" in call(launcher, "--experiment", name)


@pytest.mark.parametrize(
    "study", ["provenance_prepare", "sampling_audit", "sampling_precision"]
)
def test_native_grouped_job_uses_selected_row(study, monkeypatch, tmp_path, capsys):
    from dataclasses import replace
    from importlib import import_module
    import shlex

    provenance = study == "provenance_prepare"
    family = "instruct-response-provenance" if provenance else "paper-evidence-followup"
    package = "provenance" if provenance else "serving"
    config = import_module("experiments." + study)
    native = import_module(f"workloads.{package}.{study}")
    table = "job_configs" if hasattr(config, "job_configs") else "experiment_configs"
    original = getattr(config, table)[0]
    changes = {"name": original.name + "-second"}
    if provenance:
        changes["directory"] = "prepared-second"
    second = replace(original, **changes)
    for module in (config, native):
        monkeypatch.setattr(module, table, [original, second])
    monkeypatch.setattr(native, "ROOT", tmp_path)
    run = family_runs(family)[second.name, "prepare" if provenance else "eval"]
    monkeypatch.setattr(sys, "argv", [run.script, *run.args, "--dry"])
    native.main()
    preview = json.loads(capsys.readouterr().out)
    output = tmp_path / ("prepared-second" if provenance else "runs/out/" + second.name)
    assert preview["output_dir"] == str(output)
    command = shlex.split(shlex.split(preview["payload"]["script"])[-1])
    args = command[command.index(run.script) + 1 :]
    from experiments.family_cli import PREFIX

    monkeypatch.setattr(
        sys, "argv", [run.script, *(arg for arg in args if not arg.startswith(PREFIX))]
    )
    seen = []
    monkeypatch.setattr(native, "worker", lambda job: seen.append(job))
    native.main()
    assert seen == [second]


def test_shared_worker_matches_exact_name_when_another_has_same_prefix(monkeypatch):
    from dataclasses import replace
    from orchestration import experiment_suite as suite

    original = suite.experiment_configs[0]
    monkeypatch.setattr(
        suite,
        "experiment_configs",
        [original, replace(original, name=original.name + "-v2")],
    )
    assert suite.collect_experiments(original.name) == [original]


def test_vllm_trainer_receives_selected_run(monkeypatch, tmp_path):
    import argparse
    from dataclasses import replace
    from unittest.mock import Mock
    from experiments import vllm_sps_config as config
    from workloads.serving import vllm_sps_run as launcher, vllm_sps_train as trainer

    original = config.collect_experiments("train")[0]
    second = replace(original, name=original.name + "-second", num_gpus=1)
    monkeypatch.setattr(config, "experiment_configs", [original, second])
    monkeypatch.setattr(config.family_config, "ROOT", tmp_path)
    popen = Mock()
    popen.return_value.wait.return_value = 0
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    launcher.worker(argparse.Namespace(phase="train", filter=second.name))
    command = popen.call_args.args[0]
    script = command.index("scripts/serving/vllm_sps_train.py")
    assert "--nproc_per_node=1" in command
    monkeypatch.setattr(sys, "argv", command[script:])
    for key, value in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}.items():
        monkeypatch.setenv(key, value)

    class SelectedAllocation(Exception):
        pass

    monkeypatch.setattr(
        trainer.torch.cuda, "set_device", Mock(side_effect=SelectedAllocation)
    )
    with pytest.raises(SelectedAllocation):
        trainer.main()
    assert second.output("train") == tmp_path / "runs/out" / second.name
    assert second.output("probe") == tmp_path / "sanity/out" / second.name


def test_serving_replicates_with_same_arm_select_one_worker(monkeypatch, capsys):
    from dataclasses import replace
    import shlex
    from experiments import real_readout_serving as config
    from workloads.readout import real_readout_serving as native

    original = config.experiment_configs[0]
    replicate = replace(original, name="replicate-A")
    for module in (config, native):
        monkeypatch.setattr(module, "experiment_configs", [original, replicate])
    run = family_runs("qwen3-instruct-real-readout-mtp")["serving-replicate-A", "eval"]
    monkeypatch.setattr(sys, "argv", [run.script, *run.args, "--dry"])
    native.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["output_dir"].endswith("/replicate-A")
    cmd = shlex.split(shlex.split(plan["payload"]["script"])[-1])
    args = [v for v in cmd[cmd.index(run.script):] if not v.startswith("--cfg.")]
    seen = []
    monkeypatch.setattr(native, "worker", seen.append)
    monkeypatch.setattr(sys, "argv", args)
    native.main()
    assert seen == [replicate]
