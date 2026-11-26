"""Submit shared MLSpace phases with the configured result directory as their identity."""

from contextlib import nullcontext
import json
import shlex
from pathlib import Path

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
    return destination("prepare").output / "prepared.json"


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


def pilot_preview(run, root, phase):
    from orchestration.experiment_suite import (
        TASKS,
        DATASET_REPO,
        DATASET_REVISION,
        DATA_FILES,
        eval_command,
        train_command,
    )

    work = destination(phase, run).work
    probe = phase == "prepare"
    data_root = work if probe else root
    result = run.config(data_root, probe=probe)
    result["data_source"] = dict(
        repo=DATASET_REPO, revision=DATASET_REVISION, files=DATA_FILES
    )
    if phase == "eval":
        marker = run.output_dir / "trained.json"
        checkpoint = Path(json.loads(marker.read_text())["checkpoint"]) if marker.is_file() else None
        # The filename is only known after training has published the checkpoint.
        stem = checkpoint.stem if checkpoint else "<checkpoint>"
        result["worker_commands"] = [eval_command(run, task, checkpoint=checkpoint,
            result=work / stem / f"{task}.json") for task in TASKS]
    else:
        dest = work / "out" / run.name if probe else work
        result["worker_commands"] = [train_command(run, probe=probe, data_root=data_root, dest=dest)]
    return result


def submit(args):
    from orchestration import mlspace_launch as launch

    phase = args.phase
    runs = [
        e
        for e in suite.collect_experiments()
        if (not args.filter or args.filter in e.name)
        and (not args.named or ("data-and-probes" if phase == "prepare" else e.name) == args.named)
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
                from experiments.family_cli import worker_command, worker_settings

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
                settings = worker_settings(schemas, name)
                body["script"] = worker_command(body["script"], settings)
            launch.scope_payload(body, directory.output)
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
                    if suite.SUITE in {"1b", "10b"}:
                        settings = {e.name: pilot_preview(e, suite.RUN_ROOT, phase) for e in (
                            suite.collect_experiments() if run is None else [run])}
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
            from orchestration.named.publication import retryable
            from mls.manager.job.utils import training_job_api_from_profile

            if directory.reservation.exists() and not retryable(
                directory, args.profile
            ):
                job = directory.job_name()
                if job:
                    jobs.append(dict(name=name, job_name=job, reused=True))
                continue
            client, extra = training_job_api_from_profile(args.profile)
            launch.disable_submission_retries(client)
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

            def reserve_budget():
                if suite.SUITE in suite.AR_SPS_SUITES:
                    from workloads.adaptation.ar_sps_run import reserve

                    with launch.submission_lock():
                        reserve(phase, name)
                elif suite.SUITE == suite.INSTRUCT_SUITE:
                    from workloads.instruction.instruct_run import reserve

                    with launch.submission_lock():
                        reserve(phase, name)

            jobs.append(
                launch.submit_payload(
                    client,
                    body,
                    phase,
                    name,
                    commit,
                    directory=directory,
                    before_send=reserve_budget,
                )
            )
    if args.dry and args.named:
        return 0
    print("__MLS_JOBS_JSON__")
    print(json.dumps(dict(phase=phase, jobs=jobs)))
    return 0
