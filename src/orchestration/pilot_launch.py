"""Submit shared MLSpace phases with the configured result directory as their identity."""

from contextlib import nullcontext
import hashlib
import json
import shlex
from pathlib import Path
from uuid import uuid4

from experiments.phase_outputs import families as phase_outputs
from experiments.evaluation_outputs import families as evaluation_outputs
from orchestration.run_directory import RunDirectory
from orchestration import experiment_suite as suite


def destination(phase, run=None):
    if suite.SUITE in phase_outputs:
        output = phase_outputs[suite.SUITE].path(
            suite.ARTIFACTS_ROOT, phase, run.name if run else None
        )
    elif suite.SUITE in evaluation_outputs:
        if phase != "eval":
            raise ValueError(f"{suite.SUITE} is evaluation-only")
        output = evaluation_outputs[suite.SUITE].for_run(suite.COMMON_ROOT, run)
    elif phase == "prepare":
        from experiments import pilot_1b, pilot_10b

        family = pilot_10b if suite.SUITE == "10b" else pilot_1b
        output = family.experiment_configs[0].prepare_output_dir
        if any(exp.prepare_output_dir != output for exp in family.experiment_configs):
            raise ValueError("Family preparation requires a shared prepare_output_dir")
    elif phase == "train":
        output = run.output_dir
    else:
        output = run.eval_output_dir
    root = (
        suite.COMMON_ROOT
        if suite.ARTIFACTS_ROOT.is_relative_to(suite.COMMON_ROOT)
        else suite.ARTIFACTS_ROOT
    )
    return RunDirectory(
        root if output.is_relative_to(root) else suite.COMMON_ROOT, output
    )


def prepare_report():
    path = destination("prepare").output / "prepared.json"
    if path.is_file() or path.parent != suite.ARTIFACTS_ROOT / "prepared":
        return path
    historic = suite.ARTIFACTS_ROOT / "prepared.json"
    if historic.is_file() and suite.SUITE in phase_outputs:
        return historic
    if historic.is_file():
        forecasts = json.loads(historic.read_text()).get("estimates", {})
        if all(run.name in forecasts for run in suite.collect_experiments()):
            return historic
    return path


def validate_forecast(run, report):
    forecast = report.get("estimates", {}).get(run.name)
    if (
        not forecast
        or forecast["token_budget"] != run.training.max_tokens
        or forecast["projected_training_hours"] > suite.MAX_HOURS
    ):
        raise RuntimeError(
            f"Select a new prepare_output_dir and run preparation for {run.name}"
        )
    return forecast


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
        if (not args.named or e.name == args.named)
        and (phase != "train" or e.trainable)
    ]
    if suite.SUITE in evaluation_outputs and phase != "eval":
        raise ValueError(f"{suite.SUITE} is evaluation-only")
    if not runs:
        raise ValueError("No matching experiments")
    commit = launch.git("rev-parse", "HEAD")
    staged = launch.STAGING_ROOT / commit
    jobs = []
    for run in [None] if phase == "prepare" else runs:
        name = "data-and-probes" if run is None else run.name
        directory = destination(phase, run)
        if not args.dry and directory.output.is_dir():
            print(f"Skip existing output: {directory.output}")
            continue
        guard = nullcontext(True) if args.dry else directory.submitting()
        with guard as ready:
            if not ready:
                print(f"Skip concurrent or completed submission: {directory.output}")
                continue
            body = launch.payload(
                phase,
                name,
                staged,
                args.author_name,
                args.telegram_nick,
                training_gpus=max(e.num_gpus for e in suite.collect_experiments())
                if run is None
                else run.num_gpus,
            )
            if suite.SUITE in phase_outputs:
                cd, command = body["script"].split(" && ", 1)
                wrapper = shlex.join(
                    [str(suite.PYTHON), "-m", "orchestration.phase_worker", phase, name]
                )
                body["script"] = f"{cd} && {wrapper} {command}"
            else:
                body["script"] += " --reserved"
            if suite.SUITE not in {"1b", "10b"}:
                from experiments.family_cli import worker_command

                if suite.SUITE in suite.AR_SPS_SUITES:
                    schemas = ("ar_sps_config",)
                elif suite.SUITE == suite.INSTRUCT_SUITE:
                    schemas = ("instruct_config",)
                elif suite.SUITE.startswith("mmlu-"):
                    schemas = ("mmlu", "mmlu_config")
                else:
                    schemas = ("mtp",) + {
                        "spec-baselines": ("spec_baselines",),
                        "mtp-unified": ("mtp_unified",),
                    }.get(suite.SUITE, ())
                body["script"] = worker_command(body["script"], schemas, name)
            old_identity = launch.normalize(body["job_desc"])
            key = hashlib.sha256(str(directory.output.resolve()).encode()).hexdigest()
            body["job_desc"] = (
                f"{old_identity} {key} #{args.author_name} @{args.telegram_nick}"
            )
            if args.dry:
                if (
                    not args.named
                    and phase == "eval"
                    and suite.SUITE not in {"1b", "10b"}
                ):
                    marker = suite.training_marker(run)
                    if marker is not None and not marker.is_file():
                        print(f"Skip evaluation without trained checkpoint: {name}")
                        continue
                if args.named:
                    settings = {
                        e.name: effective_settings(
                            suite.SUITE, e, suite.RUN_ROOT, phase
                        )
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
            if phase == "eval" and suite.SUITE in phase_outputs:
                legacy = suite.RUN_ROOT / "out" / name
                if (
                    directory.output == suite.ARTIFACTS_ROOT / "evaluations" / name
                    and (legacy / "evaluated.json").is_file()
                ):
                    print(f"Skip existing evaluation: {legacy}")
                    continue
            from mls.manager.job.utils import (
                get_in_progress_jobs,
                training_job_api_from_profile,
            )

            client, extra = training_job_api_from_profile(args.profile)
            launch.disable_submission_retries(client)
            listed = list(get_in_progress_jobs(client_profile=args.profile))
            active = {
                launch.normalize(j["job_desc"]): j["job_name"]
                for j in listed
            }
            identity = launch.normalize(body["job_desc"])
            # Historic descriptions lacked a destination. Match only the original layout.
            historic = (
                run is None and directory.output == suite.ARTIFACTS_ROOT / "prepared"
            ) or (
                run is not None
                and directory.output
                == suite.RUN_ROOT / "out" / name / ("eval" if phase == "eval" else "")
            )
            if suite.SUITE in phase_outputs and phase == "eval":
                historic = directory.output in {
                    suite.RUN_ROOT / "out" / name / "eval",
                    suite.ARTIFACTS_ROOT / "evaluations" / name,
                }
            if suite.SUITE in evaluation_outputs:
                historic = directory.output == suite.RUN_ROOT / "out" / name / (
                    "full" if suite.SUITE.startswith("mmlu-") else ""
                )
            job = active.get(identity)
            if historic and not job:
                job = historical_job(suite.ARTIFACTS_ROOT, old_identity, client, active)
            if directory.reservation.exists():
                attempt_file = directory.reservation / "attempt.json"
                if attempt_file.is_file():
                    attempt = json.loads(attempt_file.read_text())["attempt"]
                    job = next(
                        (
                            row["job_name"]
                            for row in listed
                            if launch.normalize(row["job_desc"]) == identity
                            and f"#attempt-{attempt}" in row["job_desc"].split()
                        ),
                        None,
                    )
                job = directory.job_name() or job
                if job and not directory.job_name():
                    directory.record(job)
                if job:
                    state = str(client.get_job_status(job).get("status", "")).lower()
                    if (
                        state in {"completed", "succeeded"}
                        and not directory.output.is_dir()
                    ):
                        raise RuntimeError(
                            f"Completed job {job} did not publish {directory.output}"
                        )
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
            if phase in {"train", "eval"} and suite.SUITE in suite.AR_SPS_SUITES | {
                suite.INSTRUCT_SUITE
            }:
                report = prepare_report()
                if not report.is_file():
                    raise RuntimeError("Complete --phase prepare first")
            if phase == "train":
                report = prepare_report()
                if not report.is_file():
                    raise RuntimeError("Complete --phase prepare first")
                if suite.SUITE not in phase_outputs:
                    validate_forecast(run, json.loads(report.read_text()))
            if (
                phase == "eval"
                and suite.SUITE in evaluation_outputs.keys() | phase_outputs.keys()
            ):
                marker = suite.training_marker(run)
                if marker is not None and not marker.is_file():
                    print(f"Skip evaluation without trained checkpoint: {name}")
                    continue
            elif phase == "eval" and suite.SUITE not in phase_outputs:
                marker = run.output_dir / "trained.json"
                if not marker.is_file():
                    print(f"Skip evaluation without trained checkpoint: {name}")
                    continue
                checkpoint = Path(json.loads(marker.read_text())["checkpoint"])
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
            launch.stage_repo(Path(launch.git("rev-parse", "--show-toplevel")), commit)
            body["region"] = extra["region"]
            if not directory.reserve():
                continue
            try:
                attempt = uuid4().hex
                body["job_desc"] += f" #attempt-{attempt}"
                launch.store_json(directory.reservation / "attempt.json", {"attempt": attempt})
                if suite.SUITE in suite.AR_SPS_SUITES:
                    from workloads.adaptation.ar_sps_run import reserve

                    with launch.submission_lock():
                        reserve(phase, name)
                elif suite.SUITE == suite.INSTRUCT_SUITE:
                    from workloads.instruction.instruct_run import reserve

                    with launch.submission_lock():
                        reserve(phase, name)
            except Exception:
                directory.retire()
                raise
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
