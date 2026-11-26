"""Launch the isolated DFlash-b16 reproduction probe, training, or evaluation."""

import argparse
import json
from pathlib import Path
import shlex
from unittest.mock import patch

from scripts import dflash_repro_eval as evaluation, dflash_repro_train as training
from scripts import mlspace_launch as launch, qwen4b_drafter_train as previous

from src.experiments.dflash_repro import TIMEOUTS


def final_step():
    return training.total_steps(len(previous.selection()["train"]))


def arm_name(phase, step):
    if phase != "eval":
        return training.EXPERIMENT.name
    identity = evaluation.signature()[:12]
    return f"gsm8k-step-{step:05d}-{identity}"


def marker(phase, step):
    if phase == "probe":
        return training.run_root() / "prepared.json"
    name = "trained.json" if phase == "train" else "evaluated.json"
    return training.run_root() / "runs" / "out" / arm_name(phase, step) / name


def payload(phase, step, staged):
    name = arm_name(phase, step)
    if phase == "eval":
        body = launch.payload("eval", name, staged, "d.tarasov", "mrsndmn")
        distributed = []
    else:
        body = launch.payload("train", name, staged, "d.tarasov", "mrsndmn")
        body["instance_type"] = launch.INSTANCE_TYPES_BY_NUM_GPUS[training.EXPERIMENT.num_gpus]
        distributed = ["-m", "torch.distributed.run", "--standalone",
                       f"--nproc-per-node={training.EXPERIMENT.num_gpus}"]
        body["env_variables"]["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    entry = ["-m", "scripts.dflash_repro", "--worker", "--phase", phase]
    if phase == "eval":
        entry += ["--step", str(step)]
    command = ["timeout", "--kill-after=60s", TIMEOUTS[phase], str(launch.PYTHON), *distributed, *entry]
    body["script"] = f"cd {shlex.quote(str(staged))} && " + shlex.join(command)
    body["job_desc"] = f"dflash-repro {phase} {name} {training.signature()[:16]} #d.tarasov @mrsndmn"
    return body


def submit(phase, step, dry):
    commit = launch.git("rev-parse", "HEAD")
    staged = launch.STAGING_ROOT / commit
    body = payload(phase, step, staged)
    if dry:
        print(json.dumps(body, indent=2))
        return
    if phase == "train" and not marker("probe", None).is_file():
        raise RuntimeError("Complete the DFlash GPU probe before training")
    if phase == "eval" and not (training.checkpoint_path(step) / "weights.safetensors").is_file():
        raise RuntimeError(f"Missing DFlash checkpoint at step {step}")
    submit_job(phase, arm_name(phase, step), training.run_root(), marker(phase, step), body, commit)


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
            active.update({launch.normalize(job["job_desc"]): job["job_name"]
                           for job in response["jobs"]})
        launch.reconcile_receipts(client, active)
        launch.reconcile_intents(active)
        identity = launch.normalize(body["job_desc"])
        if identity in active:
            result = {"name": name, "job_name": active[identity], "reused": True}
        else:
            launch.stage_repo(Path(launch.git("rev-parse", "--show-toplevel")), commit)
            receipt_phase = "prepare" if phase == "probe" else phase
            result = launch.submit_payload(client, body, receipt_phase, name, commit)
        print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(TIMEOUTS), default="probe")
    parser.add_argument("--step", type=int, help="Select a saved checkpoint for eval")
    modes = parser.add_mutually_exclusive_group()
    for flag in ("dry", "worker", "report"):
        modes.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    step = args.step
    if args.phase == "eval":
        last = final_step()
        step = last if step is None else step
        if step not in (*training.MILESTONES, last):
            parser.error("Choose a scheduled DFlash checkpoint")
    elif step is not None:
        parser.error("--step is only available for evaluation")

    result = marker(args.phase, step)
    if args.report:
        report = json.loads(result.read_text())
        print(json.dumps(report["summary"] if args.phase == "eval" else report, indent=2))
    elif args.worker:
        if result.is_file():
            return
        if args.phase == "eval":
            evaluation.evaluate(training.checkpoint_path(step), result)
        else:
            training.train(probe=args.phase == "probe")
    else:
        submit(args.phase, step, args.dry)


if __name__ == "__main__":
    main()
