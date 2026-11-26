"""Shared MLSpace launcher for instruct reruns and continuations."""

import argparse
import json
import os
import shlex
import signal
import subprocess
import time
from importlib import import_module
from pathlib import Path

from scripts import mlspace_launch as launch


class Rerun:
    def __init__(self, family, common, config, train_variant, *, continuation=False):
        self.family = family
        self.sc = common
        self.c = config
        self.train_variant = train_variant
        self.continuation = continuation
        self.module = f"scripts.{family}_run"

    def experiments(self, pattern=None):
        return [x for x in self.sc.e.collect_experiments(pattern) if x.trainable]

    def payload(self, variant, phase, exp, workload, staged, author, nick):
        body = launch.payload(phase, exp.name, staged, author, nick)
        command = [
            str(self.sc.e.PYTHON),
            "-m",
            self.module,
            "--worker",
            "--variant",
            variant,
            "--phase",
            phase,
            "--filter",
            exp.name,
        ]
        if workload:
            command += ["--workload", workload]
        body["script"] = f"cd {shlex.quote(str(staged))} && " + shlex.join(command)
        step = f" step{variant}" if self.continuation else ""
        body["job_desc"] = (
            f"sps-{self.c.ROOT_NAMES[variant]} {phase} {self.sc.arm_name(exp, workload)}{step} #{author} @{nick}"
        )
        return body

    def hours(self, phase, workload):
        return (
            self.c.TRAIN_HOURS if phase == "train" else self.c.WORKLOAD_HOURS[workload]
        )

    def reserve(self, variant, phase, name, workload):
        path = self.sc.root(variant) / "budget.json"
        ledger = json.loads(path.read_text()) if path.exists() else {"attempts": []}
        cost = self.hours(phase, workload) * (4 if phase == "train" else 1)
        if (
            sum(x["gpu_hours"] for x in ledger["attempts"]) + cost
            > self.c.BUDGETS[variant]
        ):
            raise RuntimeError("Rerun GPU-hour reservation cap exceeded")
        ledger["attempts"].append(
            {"phase": phase, "name": name, "gpu_hours": cost, "time": time.time()}
        )
        launch.store_json(path, ledger)

    def submit(self, args):
        exps = self.experiments(args.filter)
        if not exps:
            raise ValueError("No matching trainable experiments")
        commit = launch.git("rev-parse", "HEAD")
        staged = self.sc.e.STAGING_ROOT / commit
        active = {}
        client = None
        extra = {}
        workloads = (
            [None]
            if args.phase == "train"
            else ([args.workload] if args.workload else self.c.WORKLOADS)
        )
        if not args.dry:
            self.sc.source_plan()
            if self.continuation:
                for exp in exps:
                    self.sc.validate_checkpoint(
                        args.variant, exp, source=args.phase == "train"
                    )
            elif args.phase == "eval":
                for exp in exps:
                    self.sc.validate_checkpoint(args.variant, exp)
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
        jobs = []
        for exp in exps:
            for workload in workloads:
                name = self.sc.arm_name(exp, workload)
                if self.continuation and workload:
                    name += f"--step{args.variant}"
                marker = self.sc.output_dir(args.variant, exp, workload) / (
                    "trained.json" if args.phase == "train" else "evaluated.json"
                )
                if marker.exists():
                    if (
                        json.loads(marker.read_text())["design_signature"]
                        != self.sc.signature()
                    ):
                        raise RuntimeError("Stale rerun completion marker")
                    print("Skip completed:", name)
                    continue
                body = self.payload(
                    args.variant,
                    args.phase,
                    exp,
                    workload,
                    staged,
                    args.author_name,
                    args.telegram_nick,
                )
                identity = launch.normalize(body["job_desc"])
                if identity in active:
                    jobs.append(
                        {"name": name, "job_name": active[identity], "reused": True}
                    )
                    continue
                if args.dry:
                    print(json.dumps(body, indent=2))
                    continue
                body["region"] = extra["region"]
                self.reserve(args.variant, args.phase, name, workload)
                job = launch.submit_payload(client, body, args.phase, name, commit)
                jobs.append(job)
                active[identity] = job["job_name"]
                print(json.dumps(job), flush=True)
        print("__MLS_JOBS_JSON__")
        print(
            json.dumps({"phase": args.phase, "variant": args.variant, "jobs": jobs}),
            flush=True,
        )

    def worker(self, args):
        exps = [x for x in self.experiments() if x.name == args.filter]
        if len(exps) != 1:
            raise ValueError("Select exactly one experiment")
        exp = exps[0]
        if args.phase == "eval" and not args.workload:
            raise ValueError("Select one eval workload")
        if args.internal:
            if args.phase != "eval":
                raise ValueError("Training uses torchrun")
            evaluator = import_module(f"scripts.{self.family}_eval")
            return evaluator.evaluate(args.variant, exp, args.workload, args.limit)
        self.sc.source_plan()
        if args.phase == "train":
            command = [
                str(self.sc.e.PYTHON),
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={exp.num_gpus}",
                "-m",
                f"scripts.{self.family}_train",
                "--filter",
                exp.name,
            ]
        else:
            command = [
                str(self.sc.e.PYTHON),
                "-m",
                self.module,
                "--worker",
                "--internal",
                "--variant",
                args.variant,
                "--phase",
                "eval",
                "--filter",
                exp.name,
                "--workload",
                args.workload,
            ]
            if args.limit:
                command += ["--limit", str(args.limit)]
        process = subprocess.Popen(command, start_new_session=True)
        try:
            code = process.wait(timeout=self.hours(args.phase, args.workload) * 3600)
            if code:
                raise RuntimeError(f"Rerun worker exited {code}")
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise

    def main(self):
        parser = argparse.ArgumentParser()
        variant_options = (
            {"default": self.train_variant} if self.continuation else {"required": True}
        )
        parser.add_argument(
            "--variant", choices=tuple(self.c.ROOT_NAMES), **variant_options
        )
        parser.add_argument("--phase", choices=("train", "eval"), required=True)
        parser.add_argument("--workload", choices=self.c.WORKLOADS)
        parser.add_argument("--filter")
        parser.add_argument("--worker", action="store_true")
        parser.add_argument("--internal", action="store_true")
        parser.add_argument("--limit", type=int)
        parser.add_argument("--dry", action="store_true")
        parser.add_argument("--profile", default="default")
        parser.add_argument("--author_name", default="d.tarasov")
        parser.add_argument("--telegram_nick", default="mrsndmn")
        args = parser.parse_args()
        if args.phase == "train" and (
            args.variant != self.train_variant or args.workload or args.limit
        ):
            parser.error(
                f"Training requires variant {self.train_variant}, without eval workload or limit"
            )
        if args.limit is not None and args.limit < 1:
            parser.error("Positive limit required")
        if args.worker:
            return self.worker(args)
        if args.internal or args.limit:
            parser.error("--internal/--limit require --worker")
        launch.ARTIFACTS_ROOT = self.sc.root(args.variant)
        launch.RUN_ROOT = launch.ARTIFACTS_ROOT / "runs"
        if args.dry:
            return self.submit(args)
        with launch.submission_lock():
            return self.submit(args)
