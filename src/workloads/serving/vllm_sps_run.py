"""Isolated, deduplicated MLSpace SPS-MTP training and probe jobs."""
from experiments.family_cli import worker_child

import argparse
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

from orchestration import mlspace_launch as launch
from workloads.serving import vllm_sps_common as common
from experiments import vllm_sps_config as c


def payload(phase, exp, staged, author, nick):
    if phase not in ("probe", "train"):
        raise ValueError("Use serving_benchmark for vLLM evaluation")
    body = launch.payload("train", exp.name, staged, author, nick)
    body["job_desc"] = f"sps-vllm-qwen3 {phase} {exp.name} #{author} @{nick}"
    body["instance_type"] = launch.INSTANCE_TYPES_BY_NUM_GPUS[exp.num_gpus]
    body["script"] = f"cd {shlex.quote(str(staged))} && " + shlex.join(
        [
            str(launch.PYTHON),
            "scripts/serving/vllm_sps_run.py",
            "--worker",
            "--phase",
            phase,
            "--filter",
            exp.name,
        ]
    )
    body["env_variables"].update(
        HF_HUB_DISABLE_XET="1", SPS_EXPERIMENT_SUITE="ar-sps-qwen3-1.7b-instruct"
    )
    return body


def reserve(phase, exp):
    path = c.family_config.ROOT / "budget.json"
    ledger = json.loads(path.read_text()) if path.exists() else {"attempts": []}
    hours = {"probe": c.family_config.PROBE_HOURS, "train": c.family_config.TRAIN_HOURS, "eval": c.family_config.EVAL_HOURS}[
        phase
    ] * exp.num_gpus
    if (
        sum(x["gpu_hours"] for x in ledger["attempts"]) + hours
        > c.family_config.RESERVATION_CAP_GPU_HOURS
    ):
        raise RuntimeError("SPS-vLLM GPU-hour reservation cap exceeded")
    ledger["attempts"].append(
        {
            "phase": phase,
            "experiment": exp.name,
            "gpu_hours": hours,
            "time": time.time(),
        }
    )
    launch.store_json(path, ledger)


def submit(args):
    experiments = c.collect_experiments(args.phase, args.filter)
    if not experiments:
        raise ValueError("No matching experiments")
    commit = launch.git("rev-parse", "HEAD")
    staged = launch.STAGING_ROOT / commit
    if not args.dry:
        experiments = [exp for exp in experiments if launch.pending(exp.output(args.phase), args.profile)]
        if not experiments:
            return
    batch = launch.SubmissionBatch(commit, args.profile)
    jobs = []
    for exp in experiments:
        root = exp.output(args.phase)
        if not args.dry and root.is_dir():
            continue
        if not args.dry and args.phase == "train":
            estimate = common.probe_seconds(exp.output("probe") / "probe.json", num_gpus=exp.num_gpus)
            if estimate * 2 > c.family_config.TRAIN_HOURS * 3600:
                raise RuntimeError("Training exceeds bounded reservation")
        body = payload(args.phase, exp, staged, args.author_name, args.telegram_nick)
        if args.dry:
            launch.print_preview(body, root, staged, args.phase)
            continue
        job = batch.submit(body, "prepare" if args.phase == "probe" else args.phase, exp.name, root,
            before_send=lambda: reserve(args.phase, exp))
        jobs.append(job)
        print(json.dumps(job), flush=True)
    if args.dry:
        return
    print("__MLS_JOBS_JSON__")
    print(json.dumps({"phase": args.phase, "jobs": jobs}), flush=True)


def worker(args):
    matches = c.collect_experiments(args.phase, args.filter)
    if len(matches) != 1 or matches[0].name != args.filter:
        raise ValueError("Select exactly one experiment by complete name")
    command = [
        str(launch.PYTHON),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={matches[0].num_gpus}",
        "scripts/serving/vllm_sps_train.py", "--filter", args.filter,
    ]
    if args.phase == "probe":
        command += ["--probe"]
    process = subprocess.Popen(worker_child(command), start_new_session=True)
    try:
        hours = c.family_config.PROBE_HOURS if args.phase == "probe" else c.family_config.TRAIN_HOURS
        code = process.wait(timeout=hours * 3600)
        if code:
            raise RuntimeError(f"SPS MTP worker exited {code}")
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("probe", "train"), required=True)
    parser.add_argument("--filter")
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--author_name", default="d.tarasov")
    parser.add_argument("--telegram_nick", default="mrsndmn")
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    launch.ARTIFACTS_ROOT = c.family_config.ROOT
    launch.RUN_ROOT = c.family_config.ROOT / "runs"
    if args.dry:
        return submit(args)
    with launch.submission_lock():
        return submit(args)
