from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from orchestration import mlspace_launch, pilot_launch
from orchestration.run_directory import RunDirectory


def test_concurrent_reservation_and_atomic_publish(tmp_path):
    run = RunDirectory(tmp_path, tmp_path / "runs" / "example")
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(lambda _: run.reserve(), range(24))) == 1
    run.work.mkdir()
    (run.work / "metric").write_text("0.75")
    assert not run.output.exists()
    run.finish()
    assert (run.output / "metric").read_text() == "0.75"
    assert not run.reservation.exists()
    assert not run.reserve()


def test_submission_lifecycle(tmp_path, monkeypatch):
    run = RunDirectory(tmp_path, tmp_path / "prepared")
    job = "lm-mpi-job-48fcb78d-e22e-4076-a28d-f92f4e1d6b0c"
    states = []
    client = SimpleNamespace(status="Pending")

    def send(body):
        states.append(body)
        assert run.reservation.is_dir()
        return {"job_name": job}

    client.get_list_jobs = lambda *args: {"jobs": []}
    client.run_job = send
    client.get_job_status = lambda _: {"status": client.status}
    api = ModuleType("mls.manager.job.utils")
    api.training_job_api_from_profile = lambda _: (client, {"region": "test"})
    api.get_in_progress_jobs = lambda **_: []
    monkeypatch.setitem(sys.modules, api.__name__, api)
    monkeypatch.setattr(mlspace_launch, "disable_submission_retries", lambda _: None)
    monkeypatch.setattr(mlspace_launch, "stage_repo", lambda *args: None)
    monkeypatch.setattr(pilot_launch, "destination", lambda *args: run)
    monkeypatch.setattr(pilot_launch.suite, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(mlspace_launch, "ARTIFACTS_ROOT", tmp_path)
    args = SimpleNamespace(
        phase="prepare",
        dry=False,
        named=None,
        filter=None,
        author_name="a",
        telegram_nick="n",
        profile="test",
    )
    assert pilot_launch.submit(args) == pilot_launch.submit(args) == 0
    assert len(states) == 1 and run.job_name() == job
    client.status = "Cancelled"
    def lost(body):
        states.append(body)
        raise TimeoutError("response lost")
    client.run_job = lost
    with pytest.raises(TimeoutError):
        pilot_launch.submit(args)
    assert len(states) == 2
    api.get_in_progress_jobs = lambda **_: [dict(job_name=job, job_desc=states[0]["job_desc"])]
    assert pilot_launch.submit(args) == 0
    assert len(states) == 2 and run.job_name() is None
    retry = "lm-mpi-job-22222222-2222-4222-8222-222222222222"
    api.get_in_progress_jobs = lambda **_: [dict(job_name=retry, job_desc=states[1]["job_desc"])]
    client.status = "Running"
    assert pilot_launch.submit(args) == 0
    assert len(states) == 2 and run.job_name() == retry
    run.work.mkdir()
    run.finish()
    assert pilot_launch.submit(args) == 0 and len(states) == 2


def test_worker_cli_round_trip(tmp_path):
    pytest.importorskip("transformers")
    from experiments.worker_args import parse_config
    from orchestration.experiment_suite import collect_experiments, pilot_cli

    for exp in collect_experiments():
        args = pilot_cli(exp, tmp_path, probe=True)
        cfg = parse_config("train", args)
        expected = exp.config(tmp_path, probe=True)
        for group in expected:
            for key, value in expected[group].items():
                assert cfg[group][key] == value


def test_named_evaluation_uses_recorded_checkpoint_and_reserved_output(
    tmp_path, monkeypatch
):
    import json
    from dataclasses import replace

    run = replace(
        pilot_launch.suite.collect_experiments()[0],
        output_dir=tmp_path / "train",
        eval_output_dir=tmp_path / "eval",
    )
    monkeypatch.setattr(pilot_launch.suite, "COMMON_ROOT", tmp_path)
    run.output_dir.mkdir()
    checkpoint = run.output_dir / "recorded_final.pt"
    checkpoint.touch()
    (run.output_dir / "trained.json").write_text(
        json.dumps({"checkpoint": str(checkpoint)})
    )
    commands = pilot_launch.pilot_preview(run, tmp_path, "eval")["worker_commands"]
    for cmd in commands:
        assert cmd[cmd.index("--checkpoint_path") + 1] == str(checkpoint)
        result = Path(cmd[cmd.index("--output_path") + 1])
        assert result.is_relative_to(pilot_launch.destination("eval", run).work)


def test_retry_recovers_partial_results_and_preserves_previous_receipt(tmp_path):
    run = RunDirectory(tmp_path, tmp_path / "mmlu")
    assert run.reserve()
    run.work.mkdir()
    (run.work / "samples.jsonl").write_text('{"index": 0}\n')
    job = "lm-mpi-job-48fcb78d-e22e-4076-a28d-f92f4e1d6b0c"
    run.record(job)
    run.retire()
    assert run.reserve()
    assert run.job_name() is None
    assert (run.failed_attempt / "job.json").is_file()
    assert (run.work / "samples.jsonl").read_text() == '{"index": 0}\n'


def test_specialized_worker_round_trips_family_settings(tmp_path, monkeypatch):
    from experiments import qwen4b_drafter_train as family
    from experiments.family_cli import parse_record

    parsed = parse_record(family.family_config, [("lr", "0.001")])
    assert parsed.LR == 0.001
    assert parsed.ROWS == family.family_config.ROWS
    assert parsed.ROOT == family.family_config.ROOT


def test_native_and_named_submission_share_the_same_reservation(tmp_path, monkeypatch):
    import json
    import shlex
    from orchestration.named import publication

    output = tmp_path / "eval"
    monkeypatch.setattr(publication, "COMMON_ROOT", tmp_path)
    monkeypatch.setattr(mlspace_launch, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setenv("SPS_FAMILY_SCHEMAS", "")
    submitted = []
    job = "lm-mpi-job-48fcb78d-e22e-4076-a28d-f92f4e1d6b0c"
    client = SimpleNamespace(run_job=lambda body: submitted.append(dict(body)) or {"job_name": job})
    body = {"job_desc": "evaluate arm #a @b", "script": "echo done"}
    native = mlspace_launch.submit_payload(client, dict(body), "eval", "arm", "commit", output_dir=output)
    assert native["job_name"] == job
    directory = RunDirectory(tmp_path, output)
    assert directory.job_name() == job
    assert not output.exists()
    assert "orchestration.named.publication" in shlex.split(submitted[0]["script"])
    assert mlspace_launch.submit_payload(client, dict(body), "eval", "arm", "commit", output_dir=output)["reused"]
    assert len(submitted) == 1
    other = tmp_path / "second"
    identity = mlspace_launch.normalize(submitted[0]["job_desc"])
    assert mlspace_launch.output_identity(body["job_desc"], output) == identity
    assert mlspace_launch.output_identity(body["job_desc"], other) != identity
    assert json.loads((directory.reservation / "submission.json").read_text())["identity"] == identity


def test_dflash_worker_uses_explicit_run_fields(monkeypatch):
    from workloads.distillation import dflash_kd_train as trainer, dflash_repro as jobs

    received = []
    monkeypatch.setattr(trainer, "train", lambda experiment, probe: received.append(experiment))
    jobs.worker(trainer, None, "probe", ["--name", "test", "--output_root", "/tmp/test",
                                         "--epochs", "7", "--temperature", "2.5"])
    assert received[0].epochs == 7 and received[0].temperature == 2.5
    assert received[0].name == "test"


@pytest.mark.parametrize("family", ["ar-sps", "ar-sps-qwen3-1.7b-instruct"])
def test_shared_submission_requires_current_preparation(tmp_path, monkeypatch, family):
    from experiments import family_cli

    suite = pilot_launch.suite
    run = SimpleNamespace(name="original", trainable=False, num_gpus=1)
    monkeypatch.setattr(suite, "SUITE", family)
    monkeypatch.setattr(suite, "COMMON_ROOT", tmp_path)
    monkeypatch.setattr(suite, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(suite, "RUN_ROOT", tmp_path / "runs")
    monkeypatch.setattr(suite, "collect_experiments", lambda *args: [run])

    monkeypatch.setattr(family_cli, "worker_command", lambda command, *args: command)
    api = ModuleType("mls.manager.job.utils")
    api.get_in_progress_jobs = lambda **kwargs: []
    api.training_job_api_from_profile = lambda profile: (SimpleNamespace(), {})
    monkeypatch.setitem(sys.modules, api.__name__, api)
    monkeypatch.setattr(mlspace_launch, "disable_submission_retries", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--filter", "original"])
    with pytest.raises(RuntimeError, match="prepare"):
        mlspace_launch.main("eval")


def test_ambiguous_retry_cannot_adopt_an_older_failed_receipt(tmp_path, monkeypatch):
    import json
    from orchestration.named import publication

    run = RunDirectory(tmp_path, tmp_path / "eval")
    job = "lm-mpi-job-48fcb78d-e22e-4076-a28d-f92f4e1d6b0c"
    monkeypatch.setattr(publication, "COMMON_ROOT", tmp_path)
    monkeypatch.setattr(mlspace_launch, "ARTIFACTS_ROOT", tmp_path)
    api = ModuleType("mls.manager.job.utils")
    api.get_in_progress_jobs = lambda **_: []
    client = SimpleNamespace(run_job=lambda body: {"job_name": job},
                             get_job_status=lambda _: {"status": "failed"})
    api.training_job_api_from_profile = lambda _: (client, {})
    monkeypatch.setitem(sys.modules, api.__name__, api)
    body = {"job_desc": "worker #owner", "script": "echo done"}
    mlspace_launch.submit_payload(client, dict(body), "eval", "arm", "head", output_dir=run.output)
    assert publication.retryable(run)
    def lost(body):
        raise ConnectionError("response lost")
    client.run_job = lost
    with pytest.raises(ConnectionError):
        mlspace_launch.submit_payload(client, dict(body), "eval", "arm", "head", output_dir=run.output)
    saved = json.loads((run.reservation / "submission.json").read_text())
    assert not publication.retryable(run)
    assert run.reservation.is_dir() and run.job_name() is None
    mlspace_launch.reconcile_for_outputs(client, {}, [tmp_path / "independent"])
    retry = "lm-mpi-job-22222222-2222-4222-8222-222222222222"
    api.get_in_progress_jobs = lambda **_: [dict(job_name=retry, job_desc=saved["identity"]+" #attempt-"+saved["attempt"])]
    client.get_job_status = lambda _: {"status": "running"}
    assert not publication.retryable(run) and run.job_name() == retry
    assert not Path(saved["intent"]).exists()


def test_native_and_named_commands_include_the_same_schedule_arguments(tmp_path, monkeypatch):
    from orchestration.named import publication
    monkeypatch.setattr(publication, "COMMON_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["scripts/instruction/sched_run.py"])
    output = tmp_path / "cosine"
    command = "echo --filter qwen3_instruct_staged_sps_ce"
    body = {"job_desc": "schedule", "script": command}
    body, settings = mlspace_launch.prepare_payload(body, output, tmp_path, "train")
    assert "--cfg.instruct_config.weight_decay" in body["script"]
    assert "weight_decay" in settings["instruct_config"]


def test_family_capacity_counts_pending_siblings_when_list_lags(tmp_path, monkeypatch):
    prefix = "sps-draft-sweep eval "
    monkeypatch.setattr(mlspace_launch, "ARTIFACTS_ROOT", tmp_path)
    first = "lm-mpi-job-48fcb78d-e22e-4076-a28d-f92f4e1d6b0c"
    second = "lm-mpi-job-22222222-2222-4222-8222-222222222222"
    statuses = {first: "Running", second: "Pending"}
    client = SimpleNamespace(get_job_status=lambda job: {"status": statuses[job]})
    for arm, job in [("a", first), ("b", second)]:
        row = dict(identity=prefix + arm, job_name=job, output_dir=str(tmp_path / arm))
        mlspace_launch.store_json(tmp_path / "submissions" / f"{job}.json", row)
    active = {}
    mlspace_launch.reconcile_for_outputs(client, active, [tmp_path / "c"])
    assert mlspace_launch.occupied_slots(client, active, prefix) == 2
    active[prefix + "a"] = first
    assert mlspace_launch.occupied_slots(client, active, prefix) == 2
    statuses[second] = "Completed"
    assert mlspace_launch.occupied_slots(client, active, prefix) == 1
    output = tmp_path / "c"
    mlspace_launch.store_json(tmp_path / "submission-intents" / "c.json",
                              dict(identity=prefix + "c", output_dir=str(output)))
    assert mlspace_launch.occupied_slots(client, active, prefix) == 2
    run = RunDirectory(tmp_path, output)
    assert run.reserve()
    with run.writing(reserved=True):
        pass
    assert mlspace_launch.occupied_slots(client, active, prefix) == 1
    active[prefix + "c"] = "lm-mpi-job-33333333-3333-4333-8333-333333333333"
    assert mlspace_launch.occupied_slots(client, active, prefix) == 2
    other = tmp_path / "graphs"
    mlspace_launch.store_json(other / "submission-intents" / "other.json",
                              dict(identity=prefix + "graph", output_dir=str(other / "graph")))
    assert mlspace_launch.occupied_slots(client, active, prefix, (tmp_path, other)) == 3
    with mlspace_launch.submission_lock(tmp_path / "serving-capacity"):
        with pytest.raises(RuntimeError, match="Another submission"):
            with mlspace_launch.submission_lock(tmp_path / "serving-capacity"):
                pass

def test_ar_worker_retains_suite_and_derives_training_rows():
    from experiments.ar_sps import configuration
    from experiments.family_cli import parse_record

    original = configuration("ar-sps-360m-kd")
    parsed = parse_record(original, [("max_steps", "21"), ("global_batch", "4")])
    assert parsed.MAX_DATA_ROWS == 84
    assert parsed.DISTILLATION and not parsed.QWEN
    assert parsed.experiment_configs == original.experiment_configs


def test_logged_child_receives_parsed_training_arguments(tmp_path, monkeypatch):
    from workloads.pilot.pilot_run import run_logged

    script = tmp_path / "scripts" / "worker.py"
    script.parent.mkdir()
    script.write_text(
        "import sys\nfrom experiments.family_cli import read_worker_arguments\n"
        "read_worker_arguments(sys.argv)\nfrom experiments.mtp import family_config\n"
        "print(family_config.MTP_LR)\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    monkeypatch.setattr(sys, "orig_argv", [sys.executable, "--cfg.mtp.mtp_lr=0.023"])
    monkeypatch.chdir(tmp_path)
    run_logged([sys.executable, "scripts/worker.py"], tmp_path / "worker.log", timeout=30)
    assert (tmp_path / "worker.log").read_text().strip() == "0.023"


def test_worker_arguments_include_paired_conditions_and_selected_arm():
    import shlex
    from experiments.family_cli import worker_command, worker_settings
    from experiments.dflash_mask_ablation import experiment_configs

    mask = shlex.split(worker_command("mask", worker_settings(["dflash_mask_ablation"])))
    for row in experiment_configs:
        assert any(f".{row.name}." in flag for flag in mask)
    serving = worker_command("serving", worker_settings(["real_readout_serving"], "A"))
    assert ".serving-A.arm=A" in serving
    assert ".serving-B." not in serving


def test_mtp_resume_validates_training_batch_and_data(monkeypatch):
    from workloads.speculative import mtp_run

    exp = SimpleNamespace(name="mtp", horizons=3, num_gpus=4)
    prepared = {"source_sha256": "source", "data": {"manifest_sha256": "corpus"}}
    def inputs():
        return mtp_run.training_inputs(exp, prepared, None)
    initial = inputs()
    monkeypatch.setattr(mtp_run.mtp.family_config, "MTP_GLOBAL_BATCH", initial["global_batch"] * 2)
    with pytest.raises(ValueError, match="training inputs"):
        mtp_run.check_resume_inputs(initial, inputs())
    monkeypatch.setattr(mtp_run.mtp.family_config, "MTP_GLOBAL_BATCH", initial["global_batch"])
    monkeypatch.setattr(mtp_run.mtp.family_config, "MTP_MAX_HORIZONS", initial["max_horizons"] + 1)
    with pytest.raises(ValueError, match="training inputs"):
        mtp_run.check_resume_inputs(initial, inputs())
    monkeypatch.setattr(mtp_run.mtp.family_config, "MTP_MAX_HORIZONS", initial["max_horizons"])
    prepared["data"]["manifest_sha256"] = "changed"
    with pytest.raises(ValueError, match="training inputs"):
        mtp_run.check_resume_inputs(initial, inputs())


def test_batch_skips_finished_active_and_deferred_before_charging(tmp_path, monkeypatch):
    from orchestration.named import publication

    monkeypatch.setattr(publication, "COMMON_ROOT", tmp_path)
    monkeypatch.setattr(mlspace_launch, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(mlspace_launch, "disable_submission_retries", lambda client: None)
    staged, charged, sent = [], [], []
    monkeypatch.setattr(mlspace_launch, "stage_repo", lambda repo, commit: staged.append(commit))
    job = "lm-mpi-job-48fcb78d-e22e-4076-a28d-f92f4e1d6b0c"
    client = SimpleNamespace(
        get_list_jobs=lambda *args: {"jobs": []},
        get_job_status=lambda name: {"status": "Running"},
        run_job=lambda body: sent.append(body) or {"job_name": job},
    )
    api = ModuleType("mls.manager.job.utils")
    api.training_job_api_from_profile = lambda profile: (client, {"region": "test"})
    monkeypatch.setitem(sys.modules, api.__name__, api)
    batch = mlspace_launch.SubmissionBatch("commit")
    def submit(name):
        return batch.submit({"job_desc": f"test {name}", "script": "worker --name " + name},
                            "train", name, tmp_path / name, settings={"name": name},
                            capacity=("test ", 1), before_send=lambda: charged.append(name))

    (tmp_path / "finished").mkdir()
    assert submit("finished")["reused"] and batch.client is None
    assert submit("first")["job_name"] == job
    assert submit("first")["reused"]
    assert submit("second")["deferred"]
    for signature in ("older-hash", "current-hash"):
        batch.active = {f"sps-draft-sweep eval second {signature}": job}
        with pytest.raises(RuntimeError, match="older design"):
            batch.submit({"job_desc": "sps-draft-sweep eval second current-hash"}, "eval", "second",
                         tmp_path / "second", legacy_prefix="sps-draft-sweep eval second ")

    batch.active.clear()
    identity = "sps-draft-sweep eval second previous"
    old_receipt = tmp_path / "submissions" / "old.json"
    mlspace_launch.store_json(old_receipt, dict(identity=identity, job_name=job))
    for recorded in ("receipt", "intent"):
        if recorded == "intent":
            old_receipt.unlink()
            mlspace_launch.store_json(tmp_path / "submission-intents" / "old.json", dict(identity=identity))
        with pytest.raises(RuntimeError, match="older design"):
            batch.submit({"job_desc": "sps-draft-sweep eval second new"}, "eval", "second",
                         tmp_path / "second", legacy_prefix="sps-draft-sweep eval second ")
    client.get_job_status = lambda name: {"status": "Failed"}
    mlspace_launch.store_json(old_receipt, dict(identity=identity, job_name=job, attempted_at=1))
    intent = tmp_path / "submission-intents" / "old.json"
    for attempted_at, expected in [(1, 0), (2, 1), (None, 1)]:
        mlspace_launch.store_json(intent, dict(identity=identity, attempted_at=attempted_at))
        assert mlspace_launch.occupied_slots(client, {}, identity, legacy_only=True) == expected
    assert len(sent) == 1 and sent[0]["region"] == "test"
    assert staged == ["commit"] and charged == ["first"]
    assert not RunDirectory(tmp_path, tmp_path / "second").reservation.exists()
