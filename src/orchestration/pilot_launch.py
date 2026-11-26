"""Submit pilot phases with the configured result directory as their identity."""

import hashlib
import json
from pathlib import Path

from orchestration.run_directory import RunDirectory
from orchestration import experiment_suite as suite


def destination(phase, run=None):
    if phase == "prepare":
        output = suite.ARTIFACTS_ROOT / "prepared"
    elif phase == "train":
        output = run.output_dir
    else:
        output = run.eval_output_dir
    return RunDirectory(suite.COMMON_ROOT, output)


def prepare_report():
    path = destination("prepare").output / "prepared.json"
    return path if path.exists() else suite.ARTIFACTS_ROOT / "prepared.json"


TERMINAL_FAILURES = {
    "failed",
    "terminated",
    "cancelled",
    "canceled",
    "deleted",
    "stopped",
}


def historical_job(root, identity, client, active):
    """Resolve pre-migration intents and receipts, including list-API lag."""
    if identity in active:
        return active[identity]
    for receipt in (root / "submissions").glob("*.json"):
        row = json.loads(receipt.read_text())
        if row["identity"] == identity:
            state = str(
                client.get_job_status(row["job_name"]).get("status", "")
            ).casefold()
            if state not in TERMINAL_FAILURES | {"completed", "succeeded"}:
                return row["job_name"]
    for intent in (root / "submission-intents").glob("*.json"):
        if json.loads(intent.read_text())["identity"] == identity:
            raise RuntimeError(f"Unresolved historical submission at {intent}")
    return None


def submit(args):
    from orchestration import mlspace_launch as launch
    from orchestration.named.registry import effective_settings

    phase = args.phase
    runs = [
        e
        for e in suite.collect_experiments(args.filter)
        if not args.named or e.name == args.named
    ]
    if not runs:
        raise ValueError("No matching experiments")
    commit = launch.git("rev-parse", "HEAD")
    staged = launch.STAGING_ROOT / commit
    jobs = []
    for run in [None] if phase == "prepare" else runs:
        name = "data-and-probes" if run is None else run.name
        directory = destination(phase, run)
        body = launch.payload(
            phase,
            name,
            staged,
            args.author_name,
            args.telegram_nick,
            training_gpus=max(e.num_gpus for e in runs)
            if run is None
            else run.num_gpus,
        )
        body["script"] += " --reserved"
        old_identity = launch.normalize(body["job_desc"])
        key = hashlib.sha256(str(directory.output.resolve()).encode()).hexdigest()
        body["job_desc"] = (
            f"{old_identity} {key} #{args.author_name} @{args.telegram_nick}"
        )
        if args.dry:
            if args.named:
                settings = {
                    e.name: effective_settings(suite.SUITE, e, suite.RUN_ROOT, phase)
                    for e in (suite.collect_experiments() if run is None else [run])
                }
                print(
                    json.dumps(
                        dict(
                            payload=body,
                            settings=settings,
                            output_dir=str(directory.output),
                        ),
                        indent=2,
                        default=str,
                    )
                )
            else:
                print(json.dumps(body, indent=2))
            continue
        if directory.output.is_dir() or (
            phase == "prepare" and prepare_report().is_file()
        ):
            print(f"Skip existing output: {directory.output}")
            continue
        from mls.manager.job.utils import (
            get_in_progress_jobs,
            training_job_api_from_profile,
        )

        client, extra = training_job_api_from_profile(args.profile)
        launch.disable_submission_retries(client)
        active = {
            launch.normalize(j["job_desc"]): j["job_name"]
            for j in get_in_progress_jobs(client_profile=args.profile)
        }
        identity = launch.normalize(body["job_desc"])
        # Historic descriptions lacked a destination. Match only the original layout.
        historic = run is None or directory.output == suite.RUN_ROOT / "out" / name / (
            "eval" if phase == "eval" else ""
        )
        job = active.get(identity)
        if historic and not job:
            job = historical_job(suite.ARTIFACTS_ROOT, old_identity, client, active)
        if directory.reservation.exists():
            job = job or directory.job_name()
            if job and not directory.job_name():
                directory.record(job)
            if job:
                state = str(client.get_job_status(job).get("status", "")).lower()
                if state in TERMINAL_FAILURES:
                    directory.retire()
                    job = None
                else:
                    jobs.append(dict(name=name, job_name=job, reused=True))
                    continue
            else:
                print(f"Skip unresolved reservation: {directory.reservation}")
                continue
        if job:
            jobs.append(dict(name=name, job_name=job, reused=True))
            continue
        if phase == "train" and not prepare_report().is_file():
            raise RuntimeError("Complete --phase prepare first")
        if phase == "eval" and not (run.output_dir / "trained.json").is_file():
            print(f"Skip evaluation without trained checkpoint: {name}")
            continue
        launch.stage_repo(Path(launch.git("rev-parse", "--show-toplevel")), commit)
        body["region"] = extra["region"]
        if not directory.reserve():
            continue
        result = client.run_job(body)
        job = launch.valid_job_name(result)
        if not job:
            if isinstance(result, dict) and result.get("error_code") == 1:
                directory.retire()
                raise RuntimeError(f"MLSpace rejected submission: {result}")
            raise RuntimeError(
                f"Unconfirmed submission; inspect {directory.reservation}"
            )
        try:
            directory.record(job)
        except FileNotFoundError:
            if not directory.output.is_dir():
                raise
        jobs.append(dict(name=name, job_name=job, commit=commit))
    print("__MLS_JOBS_JSON__")
    print(json.dumps(dict(phase=phase, jobs=jobs)))
    return 0
