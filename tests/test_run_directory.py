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
    from orchestration.named.registry import effective_settings

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
    commands = effective_settings("1b", run, tmp_path, "eval")["worker_commands"]
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
    import json
    from experiments import qwen4b_drafter_train as family
    from experiments.family_cli import arguments, encode, parse

    values = arguments(["qwen4b_drafter_train"])
    args = encode(values) + ["--qwen4b_drafter_train__lr", "0.001"]
    overrides = parse(values, args)
    monkeypatch.setenv("SPS_FAMILY_ARGUMENTS", json.dumps(overrides))
    decoded = family.configure(family.FamilyConfig)
    assert decoded.LR == 0.001
    assert decoded.ROWS == family.family_config.ROWS
    assert decoded.LENGTHS == family.family_config.LENGTHS
    assert decoded.ROOT == family.family_config.ROOT


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
    monkeypatch.setenv("SPS_SUBMIT_OUTPUT", str(output))
    assert mlspace_launch.submit_payload(client, dict(body), "eval", "arm", "commit")["reused"]
    assert len(submitted) == 1
    other = tmp_path / "second"
    identity = mlspace_launch.normalize(submitted[0]["job_desc"])
    assert mlspace_launch.output_identity(body["job_desc"], output, {identity: job}) == identity
    assert mlspace_launch.output_identity(body["job_desc"], other, {identity: job}) != identity
    assert json.loads((directory.reservation / "submission.json").read_text())["identity"] == identity


def test_dflash_worker_uses_explicit_run_fields(monkeypatch):
    import json
    from experiments import dflash_kd_train
    from experiments.family_arguments import configure_runs
    from experiments.family_cli import arguments, encode, parse

    group = "dflash_kd_train"
    exp = dflash_kd_train.EXPERIMENT
    values = arguments([group], exp.name)
    flags = encode(values)
    flags.extend(["--dflash_kd_train__experiment_configs__0__epochs", "7"])
    monkeypatch.setenv("SPS_FAMILY_ARGUMENTS", json.dumps(parse(values, flags)))
    (selected,) = configure_runs(dflash_kd_train.__name__, "experiment_configs", [exp])
    assert selected.epochs == 7
    assert selected.global_batch == exp.global_batch
    assert selected.output_root == exp.output_root


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
    monkeypatch.setattr(suite, "design_signature", lambda: "current")
    monkeypatch.setattr(family_cli, "worker_command", lambda command, *args: command)
    api = ModuleType("mls.manager.job.utils")
    api.get_in_progress_jobs = lambda **kwargs: []
    api.training_job_api_from_profile = lambda profile: (SimpleNamespace(), {})
    monkeypatch.setitem(sys.modules, api.__name__, api)
    monkeypatch.setattr(mlspace_launch, "disable_submission_retries", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--filter", "original"])
    with pytest.raises(RuntimeError, match="prepare"):
        mlspace_launch.main("eval")
    (tmp_path / "prepared.json").write_text('{"design_signature":"obsolete"}')
    with pytest.raises(RuntimeError, match="Stale"):
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
    from experiments.protocols import schemas
    from orchestration.named import publication
    monkeypatch.setattr(publication, "COMMON_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["scripts/instruction/sched_run.py"])
    output = tmp_path / "cosine"
    command = "echo --filter qwen3_instruct_staged_sps_ce"
    body = {"job_desc": "schedule", "script": command}
    preview = publication.published_command(command, output, tmp_path, schemas("sched_run", "train"))
    publication.reserve_payload(body, output, tmp_path, phase="train", intent=tmp_path/"intent")
    assert "--instruct_config__weight_decay" in body["script"]
    assert body["script"] == preview


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
    mlspace_launch.store_json(tmp_path / "submission-intents" / "c.json", dict(identity=prefix + "c"))
    assert mlspace_launch.occupied_slots(client, active, prefix) == 2
