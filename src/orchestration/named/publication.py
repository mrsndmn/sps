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


def retryable(directory):
    from orchestration.pilot_launch import TERMINAL_FAILURES

    name = directory.job_name()
    if not name:
        intent = directory.reservation / "submission.json"
        if intent.is_file():
            from mls.manager.job.utils import get_in_progress_jobs
            from orchestration.mlspace_launch import normalize

            record = json.loads(intent.read_text())
            active = {
                normalize(j["job_desc"]): j["job_name"]
                for j in get_in_progress_jobs(client_profile="default")
            }
            name = active.get(record["identity"])
            if not name:
                for path in Path(record["receipts"]).glob("*.json"):
                    row = json.loads(path.read_text())
                    if (
                        row.get("output_dir") == str(directory.output)
                        and row["identity"] == record["identity"]
                    ):
                        name = row["job_name"]
                        break
            if name:
                directory.record(name)
    if not name:
        print(f"Skip unresolved reservation: {directory.reservation}", flush=True)
        return False
    from mls.manager.job.utils import training_job_api_from_profile

    client, _ = training_job_api_from_profile("default")
    status = str(client.get_job_status(name).get("status", "")).lower()
    if status in TERMINAL_FAILURES:
        directory.retire()
        return True
    if status in {"completed", "succeeded"} and not directory.output.exists():
        raise RuntimeError(f"Completed job {name} did not publish {directory.output}")
    print(json.dumps(dict(job_name=name, reused=True)), flush=True)
    return False


def reserve_payload(body, output, staged):
    directory = RunDirectory(COMMON_ROOT, output)
    from orchestration import mlspace_launch as launch

    from experiments.family_cli import worker_command

    schemas = os.environ.get("SPS_FAMILY_SCHEMAS", "").split(",")
    command = worker_command(body["script"], tuple(filter(None, schemas)))
    body["script"] = f"cd {shlex.quote(str(staged))} && " + shlex.join(
        [str(PYTHON), "-m", "orchestration.named.publication", str(output), command]
    )
    if not directory.reserve():
        return None
    launch.store_json(
        directory.reservation / "submission.json",
        {
            "identity": launch.normalize(body["job_desc"]),
            "receipts": str(launch.ARTIFACTS_ROOT / "submissions"),
        },
    )
    return directory


def main():
    output, command = sys.argv[1:]
    directory = RunDirectory(COMMON_ROOT, Path(output))
    with directory.writing(reserved=True) as work:
        env = dict(os.environ, SPS_JOB_OUTPUT=output, SPS_JOB_WORK=str(work))
        subprocess.run(["/bin/bash", "-c", command], env=env, check=True)


if __name__ == "__main__":
    main()
