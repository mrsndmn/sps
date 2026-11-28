"""Publish a specialized worker's result under its configured directory."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from experiments.environment import COMMON_ROOT, PYTHON
from orchestration.run_directory import RunDirectory


def work_path(path):
    """Redirect only this worker's output tree. Other experiments stay readable."""
    final = os.environ.get("SPS_JOB_OUTPUT")
    work = os.environ.get("SPS_JOB_WORK")
    path = Path(path)
    if final and work and path.is_relative_to(final):
        return Path(work) / path.relative_to(final)
    return path


def worker_output(default):
    """Destination of a single-output worker whose legacy directory was inferred."""
    return Path(os.environ.get("SPS_JOB_WORK", default))


def public_path(path):
    work = os.environ.get("SPS_JOB_WORK")
    path = Path(path)
    if work and path.is_relative_to(work):
        return Path(os.environ["SPS_JOB_OUTPUT"]) / path.relative_to(work)
    return path


@contextmanager
def submitting(output):
    directory = RunDirectory(COMMON_ROOT, output)
    if output.is_dir():
        print(f"Skip existing output: {output}", flush=True)
        yield False
        return
    with directory.submitting() as ready:
        if not ready:
            print(f"Skip concurrent or completed submission: {output}", flush=True)
            yield False
        elif directory.reservation.exists() and not retryable(directory):
            yield False
        else:
            yield True


def retryable(directory, profile="default"):
    from orchestration.pilot_launch import TERMINAL_FAILURES

    name = directory.job_name()
    meta = directory.reservation / "submission.json"
    record = json.loads(meta.read_text()) if meta.is_file() else {}
    attempt = record.get("attempt")
    if not name and attempt:
        from mls.manager.job.utils import get_in_progress_jobs
        from orchestration.mlspace_launch import normalize

        name = next(
            (
                job["job_name"]
                for job in get_in_progress_jobs(client_profile=profile)
                if normalize(job["job_desc"]) == record["identity"]
                and f"#attempt-{attempt}" in job["job_desc"].split()
            ),
            None,
        )
        if not name:
            for path in Path(record["receipts"]).glob("*.json"):
                row = json.loads(path.read_text())
                if row.get("attempt") == attempt and row.get("output_dir") == str(
                    directory.output
                ):
                    name = row["job_name"]
                    break
        if name:
            directory.record(name)
    if name and attempt:
        from orchestration.mlspace_launch import store_json

        intent = Path(record["intent"])
        if intent.is_file():
            row = json.loads(intent.read_text())
            if row.get("attempt") == attempt:
                store_json(
                    Path(record["receipts"]) / f"{name}.json", dict(row, job_name=name)
                )
                intent.unlink(missing_ok=True)
    if not name:
        print(f"Skip unresolved reservation: {directory.reservation}", flush=True)
        return False
    from mls.manager.job.utils import training_job_api_from_profile

    client, _ = training_job_api_from_profile(profile)
    status = str(client.get_job_status(name).get("status", "")).lower()
    if status in TERMINAL_FAILURES:
        directory.retire()
        return True
    if status in {"completed", "succeeded"} and not directory.output.exists():
        raise RuntimeError(f"Completed job {name} did not publish {directory.output}")
    print(json.dumps(dict(job_name=name, reused=True)), flush=True)
    return False


def pending_output(output, profile="default"):
    if output.is_dir():
        print(f"Skip existing output: {output}", flush=True)
        return False
    directory = RunDirectory(COMMON_ROOT, output)
    if not directory.reservation.exists():
        return True
    with directory.submitting() as ready:
        return ready and retryable(directory, profile)


def published_command(command, output, staged):
    return f"cd {shlex.quote(str(staged))} && " + shlex.join(
        [
            str(PYTHON),
            "-m",
            "orchestration.named.publication",
            str(output),
            command,
        ]
    )


def main():
    output, command = sys.argv[1:]
    directory = RunDirectory(COMMON_ROOT, Path(output))
    if "SPS_MPI_NODES" in os.environ:
        from orchestration.named.mpi_publication import run_mpi
        return run_mpi(directory, command, os.environ)
    with directory.writing(reserved=True) as work:
        env = dict(
            os.environ,
            SPS_JOB_OUTPUT=output,
            SPS_JOB_WORK=str(work),
        )
        subprocess.run(["/bin/bash", "-c", command], env=env, check=True)
        if not any(work.iterdir()):
            raise RuntimeError(f"Worker produced no result for {output}")


if __name__ == "__main__":
    main()
