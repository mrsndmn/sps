"""Launch the isolated DFlash-b16 reproduction probe, training, or evaluation."""

import argparse
from orchestration.named.publication import submitting, work_path
import json
from pathlib import Path
import shlex
from unittest.mock import patch

from workloads.distillation import (
    dflash_repro_eval as evaluation,
    dflash_repro_train as training,
)
from orchestration import mlspace_launch as launch

from experiments.launch import run_filter
from experiments.dflash_repro import family_config as dflash_repro_settings

TIMEOUTS = dflash_repro_settings.TIMEOUTS


def marker(experiment, phase, step=None):
    filename = {
        "probe": "prepared.json",
        "train": "trained.json",
        "eval": "evaluated.json",
    }[phase]
    return work_path(experiment.output(phase, step)) / filename


def payload(experiment, phase, step, staged, script):
    name = experiment.output(phase, step).name if phase == "eval" else experiment.name
    body = launch.payload(
        "eval" if phase == "eval" else "train", name, staged, "d.tarasov", "mrsndmn"
    )
    command = ["timeout", "--kill-after=60s", TIMEOUTS[phase], str(launch.PYTHON)]
    if phase != "eval":
        body["instance_type"] = launch.INSTANCE_TYPES_BY_NUM_GPUS[experiment.num_gpus]
        body["env_variables"]["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        command += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={experiment.num_gpus}",
        ]
    command += [script, "--worker", "--phase", phase, "--filter", experiment.name]
    if phase == "eval":
        command += ["--step", str(step)]
    body["script"] = f"cd {shlex.quote(str(staged))} && " + shlex.join(command)
    study = Path(script).stem.replace("_", "-")
    body["job_desc"] = (
        f"{study} {phase} {name} {experiment.output_root.name} #d.tarasov @mrsndmn"
    )
    return body


def submit_phase(experiment, trainer, phase, step, body, commit):
    output = experiment.output(phase, step)
    with submitting(output) as ready:
        if not ready:
            return
        if phase == "train" and not marker(experiment, "probe").is_file():
            raise RuntimeError("Complete the DFlash GPU probe before training")
        if phase == "eval":
            if not marker(experiment, "train").is_file():
                raise RuntimeError(
                    "Complete DFlash training before evaluating any checkpoint"
                )
            if not (
                trainer.checkpoint_path(step, experiment) / "weights.safetensors"
            ).is_file():
                raise FileNotFoundError(trainer.checkpoint_path(step, experiment))
        name = output.name if phase == "eval" else experiment.name
        submit_job(
            phase,
            name,
            experiment.output_root,
            marker(experiment, phase, step),
            body,
            commit,
        )


def submit_job(phase, name, root, result_path, body, commit):
    from mls.manager.job.utils import training_job_api_from_profile

    with (
        patch.object(launch, "ARTIFACTS_ROOT", root),
        patch.object(launch, "RUN_ROOT", root / "runs"),
        launch.submission_lock(),
    ):
        if result_path.is_file():
            print(result_path)
            return
        client, options = training_job_api_from_profile("default")
        launch.disable_submission_retries(client)
        body["region"] = options["region"]
        active = {}
        for status in ("Pending", "Running"):
            response = client.get_list_jobs(options["region"], None, status, 1000, 0)
            if "jobs" not in response:
                raise RuntimeError(f"Cannot establish {status} jobs")
            active.update(
                {
                    launch.normalize(job["job_desc"]): job["job_name"]
                    for job in response["jobs"]
                }
            )
        launch.reconcile_for_outputs(client, active, [result_path.parent])
        identity = launch.output_identity(body["job_desc"], result_path.parent)
        if identity in active:
            result = {"name": name, "job_name": active[identity], "reused": True}
        else:
            launch.stage_repo(Path(launch.git("rev-parse", "--show-toplevel")), commit)
            receipt_phase = "prepare" if phase == "probe" else phase
            result = launch.submit_payload(
                client, body, receipt_phase, name, commit, output_dir=result_path.parent
            )
        print(json.dumps(result), flush=True)


def run(trainer, script, evaluate):
    parser = argparse.ArgumentParser(description=__doc__)
    configs = run_filter(parser, trainer.experiment_configs)
    parser.add_argument("--phase", choices=tuple(TIMEOUTS), default="probe")
    parser.add_argument("--step", type=int)
    modes = parser.add_mutually_exclusive_group()
    for flag in ("dry", "worker", "report"):
        modes.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    experiment = configs[args.filter]
    phase, step = args.phase, args.step
    if phase == "eval" and step not in experiment.milestones:
        trained = marker(experiment, "train")
        if args.dry and not trained.is_file():
            print(json.dumps({"pending_input": str(trained)}))
            return
        final = json.loads(trained.read_text())["steps"]
        if step is not None and step != final:
            parser.error("Choose a scheduled DFlash checkpoint")
        step = final
    if phase != "eval" and step is not None:
        parser.error("--step is only available for evaluation")
    result = marker(experiment, phase, step)
    if args.report:
        report = json.loads(result.read_text())
        print(json.dumps(report["summary"] if phase == "eval" else report, indent=2))
    elif args.worker:
        if result.is_file():
            return
        if phase == "eval":
            if not marker(experiment, "train").is_file():
                raise RuntimeError(
                    "Complete DFlash training before evaluating any checkpoint"
                )
            evaluate(
                trainer.checkpoint_path(step, experiment), result, experiment=experiment
            )
        else:
            trainer.train(experiment=experiment, probe=phase == "probe")
    else:
        commit = launch.git("rev-parse", "HEAD")
        staged = launch.STAGING_ROOT / commit
        body = payload(experiment, phase, step, staged, script)
        if args.dry:
            launch.print_preview(body, experiment.output(phase, step), staged, phase)
        else:
            submit_phase(experiment, trainer, phase, step, body, commit)


def main():
    run(training, "scripts/distillation/dflash_repro.py", evaluation.evaluate)
