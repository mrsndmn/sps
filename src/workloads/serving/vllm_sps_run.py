"""Isolated, deduplicated MLSpace SPS-MTP training and probe jobs."""

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
    path = c.ROOT / "budget.json"
    ledger = json.loads(path.read_text()) if path.exists() else {"attempts": []}
    hours = {"probe": c.PROBE_HOURS, "train": c.TRAIN_HOURS, "eval": c.EVAL_HOURS}[
        phase
    ] * exp.num_gpus
    if (
        sum(x["gpu_hours"] for x in ledger["attempts"]) + hours
        > c.RESERVATION_CAP_GPU_HOURS
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
    active = {}
    extra = {}
    client = None
    if not args.dry:
        experiments = [exp for exp in experiments if launch.pending(c.ROOT / ("sanity" if args.phase == "probe" else "runs") / "out" / exp.name, args.profile)]
        if not experiments:
            return
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
        launch.reconcile_for_outputs(client, active, [c.ROOT / ("sanity" if args.phase == "probe" else "runs") / "out" / exp.name for exp in experiments])
        staged = launch.stage_repo(
            Path(launch.git("rev-parse", "--show-toplevel")), commit
        )
        if args.phase == "train":
            estimate = common.probe_seconds(c.ROOT / "sanity/out/qwen3_sps_mtp/probe.json")
            if estimate * 2 > c.TRAIN_HOURS * 3600:
                raise RuntimeError("Training exceeds bounded reservation")
    jobs = []
    for exp in experiments:
        root = (
            c.ROOT / ("sanity" if args.phase == "probe" else "runs") / "out" / exp.name
        )
        marker = root / ("probe.json" if args.phase == "probe" else "trained.json")
        if marker.exists():
            if args.phase == "probe":
                common.probe_seconds(marker)
            else:
                common.trained()
            print("Skip completed:", exp.name)
            continue
        body = payload(args.phase, exp, staged, args.author_name, args.telegram_nick)
        identity = launch.output_identity(body["job_desc"], root, active)
        if identity in active:
            jobs.append(
                {"name": exp.name, "job_name": active[identity], "reused": True}
            )
            continue
        if args.dry:
            print(json.dumps(body, indent=2))
            continue
        body["region"] = extra["region"]
        reserve(args.phase, exp)
        job = launch.submit_payload(
            client,
            body,
            "prepare" if args.phase == "probe" else args.phase,
            exp.name,
            commit,
        output_dir=root)
        jobs.append(job)
        if job.get("job_name"):
            active[identity] = job["job_name"]
        print(json.dumps(job), flush=True)
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
        "scripts/serving/vllm_sps_train.py",
    ]
    if args.phase == "probe":
        command += ["--probe"]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        hours = c.PROBE_HOURS if args.phase == "probe" else c.TRAIN_HOURS
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
    launch.ARTIFACTS_ROOT = c.ROOT
    launch.RUN_ROOT = c.ROOT / "runs"
    if args.dry:
        return submit(args)
    with launch.submission_lock():
        return submit(args)
