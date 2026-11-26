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
    assert pilot_launch.submit(args) == 0
    assert len(states) == 2
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
