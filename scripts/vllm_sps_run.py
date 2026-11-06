"""Isolated, deduplicated MLSpace SPS-MTP probe/train and one-GPU eval jobs."""

import argparse
import json
import os
import shlex
import signal
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from scripts import mlspace_launch as launch
from scripts import vllm_sps_common as common
from scripts import vllm_sps_config as c


def payload(phase, exp, staged, author, nick):
    body = launch.payload(
        "eval" if phase == "eval" else "train", exp.name, staged, author, nick
    )
    body["job_desc"] = f"sps-vllm-qwen3 {phase} {exp.name} #{author} @{nick}"
    body["instance_type"] = launch.INSTANCE_TYPES_BY_NUM_GPUS[exp.num_gpus]
    body["script"] = f"cd {shlex.quote(str(staged))} && " + shlex.join(
        [
            str(launch.PYTHON),
            "-m",
            "scripts.vllm_sps_run",
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
    if phase == "eval":
        snapshot = c.ROOT / "vllm-code" / str(c.VLLM_COMMIT)
        body["env_variables"].update(
            SPS_VLLM_SOURCE=str(snapshot), VLLM_USE_V2_MODEL_RUNNER="0"
        )
        body["env_variables"]["PYTHONPATH"] = (
            str(snapshot) + ":" + body["env_variables"]["PYTHONPATH"]
        )
    return body


def stage_vllm():
    if not c.VLLM_COMMIT:
        raise RuntimeError("Pin the reviewed vLLM implementation commit first")
    if launch.git("-C", str(c.VLLM_WORKTREE), "diff", "--name-only", "HEAD"):
        raise RuntimeError("Commit vLLM changes before staging")
    target = c.ROOT / "vllm-code" / c.VLLM_COMMIT
    if target.exists():
        if (target / "source-commit.txt").read_text().strip() != c.VLLM_COMMIT:
            raise RuntimeError("vLLM snapshot identity mismatch")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        process = subprocess.Popen(
            ["git", "-C", str(c.VLLM_WORKTREE), "archive", c.VLLM_COMMIT],
            stdout=subprocess.PIPE,
        )
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            archive.extractall(temporary)
        if process.wait():
            raise RuntimeError("Cannot stage pinned vLLM commit")
        Path(temporary, "source-commit.txt").write_text(c.VLLM_COMMIT + "\n")
        Path(temporary).rename(target)
    return target


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
        if args.phase == "eval":
            trained = json.loads(
                (c.ROOT / "runs/out/qwen3_sps_mtp/trained.json").read_text()
            )
            if trained["design_signature"] != common.signature():
                raise RuntimeError("Training design mismatch")
            # Do not consume a GPU reservation with an incompatible runtime.
            import sys

            if sys.version_info < (3, 11):
                raise RuntimeError(
                    "Fresh vLLM main requires Python >=3.11; current project env is Python 3.10"
                )
            stage_vllm()
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
        launch.reconcile_receipts(client, active)
        launch.reconcile_intents(active)
        staged = launch.stage_repo(
            Path(launch.git("rev-parse", "--show-toplevel")), commit
        )
        if args.phase == "train":
            probe = json.loads(
                (c.ROOT / "sanity/out/qwen3_sps_mtp/probe.json").read_text()
            )
            estimate = probe["embedding"] * c.STAGE_STEPS + probe["full"] * (
                c.TRAIN_STEPS - c.STAGE_STEPS
            )
            if (
                probe["design_signature"] != common.signature()
                or estimate * 2 > c.TRAIN_HOURS * 3600
            ):
                raise RuntimeError(
                    "Probe is stale or training exceeds bounded reservation"
                )
    jobs = []
    for exp in experiments:
        root = (
            c.ROOT / ("sanity" if args.phase == "probe" else "runs") / "out" / exp.name
        )
        marker = (
            root
            / (
                {
                    "probe": "probe.json",
                    "train": "trained.json",
                    "eval": "evaluated.json",
                }[args.phase]
            )
        )
        if marker.exists():
            if json.loads(marker.read_text())["design_signature"] != common.signature():
                raise RuntimeError(f"Stale marker: {marker}")
            print("Skip completed:", exp.name)
            continue
        body = payload(args.phase, exp, staged, args.author_name, args.telegram_nick)
        identity = launch.normalize(body["job_desc"])
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
        )
        jobs.append(job)
        active[identity] = job["job_name"]
        print(json.dumps(job), flush=True)
    print("__MLS_JOBS_JSON__")
    print(json.dumps({"phase": args.phase, "jobs": jobs}), flush=True)


def worker(args):
    matches = c.collect_experiments(args.phase, args.filter)
    if len(matches) != 1 or matches[0].name != args.filter:
        raise ValueError("Select exactly one experiment by complete name")
    if args.phase == "eval":
        from scripts.vllm_sps_evaluate import evaluate

        # The watchdog covers evaluation too, including imports/compilation.
        command = [
            str(launch.PYTHON),
            "-m",
            "scripts.vllm_sps_run",
            "--internal-eval",
            "--phase",
            "eval",
            "--filter",
            args.filter,
        ]
        if args.internal_eval:
            return evaluate(matches[0])
    else:
        command = [
            str(launch.PYTHON),
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={matches[0].num_gpus}",
            "-m",
            "scripts.vllm_sps_train",
        ]
    if args.phase == "probe":
        command += ["--probe"]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        code = process.wait(
            timeout={
                "probe": c.PROBE_HOURS,
                "train": c.TRAIN_HOURS,
                "eval": c.EVAL_HOURS,
            }[args.phase]
            * 3600
        )
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
    parser.add_argument("--phase", choices=("probe", "train", "eval"), required=True)
    parser.add_argument("--filter")
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--internal-eval", action="store_true")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--author_name", default="d.tarasov")
    parser.add_argument("--telegram_nick", default="mrsndmn")
    args = parser.parse_args()
    if args.internal_eval and args.phase != "eval":
        parser.error("--internal-eval requires eval phase")
    if args.worker or args.internal_eval:
        return worker(args)
    launch.ARTIFACTS_ROOT = c.ROOT
    launch.RUN_ROOT = c.ROOT / "runs"
    if args.dry:
        return submit(args)
    with launch.submission_lock():
        return submit(args)


if __name__ == "__main__":
    main()
