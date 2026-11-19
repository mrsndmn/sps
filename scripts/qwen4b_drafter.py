"""Submit and report the Qwen3-4B mask and mask-embedding experiment."""

import argparse
import json
from pathlib import Path
import shlex
from unittest.mock import patch

from scripts import mlspace_launch as launch
from scripts import qwen4b_drafter_eval as evaluation
from scripts import qwen4b_drafter_train as training

PHASE_TIMEOUT = {"probe": "40m", "train": "6h", "eval": "3h"}


def root():
    return training.ROOT / training.signature()[:16]


def marker(phase, name):
    if phase == "probe":
        return root() / "prepared.json"
    return root() / "runs" / "out" / name / (
        "trained.json" if phase == "train" else "evaluated.json"
    )


def names(phase):
    if phase == "probe":
        return ["probe"]
    if phase == "eval":
        return ["eval-" + evaluation.signature()[:16]]
    return [exp.name for exp in training.experiment_configs]


def payload(phase, name, staged):
    body = launch.payload("eval", name, staged, "d.tarasov", "mrsndmn")
    body["job_desc"] = (
        f"qwen4b-drafter {phase} {name} {training.signature()[:16]} #d.tarasov @mrsndmn"
    )
    body["script"] = f"cd {shlex.quote(str(staged))} && " + shlex.join([
        "timeout", "--kill-after=60s", PHASE_TIMEOUT[phase], str(launch.PYTHON),
        "-m", "scripts.qwen4b_drafter", "--phase", phase, "--worker", "--filter", name,
    ])
    return body


def submit(phase, selected, dry):
    commit = launch.git("rev-parse", "HEAD")
    staged = launch.STAGING_ROOT / commit
    if dry:
        for name in selected:
            print(json.dumps(payload(phase, name, staged), indent=2))
        return
    if phase == "train" and not marker("probe", "probe").is_file():
        raise RuntimeError("Finish the GPU probe before submitting training")
    if phase == "eval":
        evaluation.validate_checkpoints(root() / "runs" / "out")
    with (
        patch.object(launch, "ARTIFACTS_ROOT", root()),
        patch.object(launch, "RUN_ROOT", root() / "runs"),
        launch.submission_lock(),
    ):
        return submit_pending(phase, selected, commit, staged)


def submit_pending(phase, selected, commit, staged):
    from mls.manager.job.utils import training_job_api_from_profile

    pending = [name for name in selected if not marker(phase, name).is_file()]
    if not pending:
        return
    client, options = training_job_api_from_profile("default")
    launch.disable_submission_retries(client)
    active = {}
    for status in ("Pending", "Running"):
        result = client.get_list_jobs(options["region"], None, status, 1000, 0)
        if "jobs" not in result:
            raise RuntimeError(f"Cannot establish active {status} jobs")
        active.update({launch.normalize(job["job_desc"]): job["job_name"]
                       for job in result["jobs"]})
    launch.reconcile_receipts(client, active)
    launch.reconcile_intents(active)
    launch.stage_repo(Path(launch.git("rev-parse", "--show-toplevel")), commit)
    for name in pending:
        if marker(phase, name).is_file():
            continue
        body = payload(phase, name, staged)
        body["region"] = options["region"]
        identity = launch.normalize(body["job_desc"])
        if identity in active:
            result = {"name": name, "job_name": active[identity], "reused": True}
        else:
            receipt_phase = "prepare" if phase == "probe" else phase
            result = launch.submit_payload(client, body, receipt_phase, name, commit)
        print(json.dumps(result), flush=True)


def worker(phase, name):
    import torch

    path = marker(phase, name)
    if path.is_file():
        return
    if phase == "eval":
        evaluation.evaluate(root() / "runs" / "out", path)
        return
    if torch.cuda.device_count() != 1:
        raise RuntimeError("This worker requires one visible GPU")
    if phase == "train" and not marker("probe", "probe").is_file():
        raise RuntimeError("Missing GPU probe")
    target, tokenizer = training.load_target(torch.device("cuda:0"))
    if phase == "train":
        exp = next(exp for exp in training.experiment_configs if exp.name == name)
        training.train(target, tokenizer, exp, path.parent)
        return
    reports = [training.train(target, tokenizer, exp, path.parent, probe=True)
               for exp in training.experiment_configs]
    if len({row["initial_sha256"] for row in reports}) != 1:
        raise RuntimeError("The four Transformer initializations differ")
    launch.store_json(path, reports)
    print(json.dumps(reports, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(PHASE_TIMEOUT), default="eval")
    parser.add_argument("--filter", help="Exact experiment name")
    mode = parser.add_mutually_exclusive_group()
    for option in ("worker", "dry", "report"):
        mode.add_argument(f"--{option}", action="store_true")
    args = parser.parse_args()
    selected = [name for name in names(args.phase) if args.filter in (None, name)]
    if not selected:
        parser.error("No matching experiment")
    if args.report:
        for name in selected:
            report = json.loads(marker(args.phase, name).read_text())
            print(json.dumps(report["summary"] if args.phase == "eval" else report, indent=2))
    elif args.worker:
        if len(selected) != 1:
            parser.error("A worker runs one experiment")
        worker(args.phase, selected[0])
    else:
        submit(args.phase, selected, args.dry)


if __name__ == "__main__":
    main()
