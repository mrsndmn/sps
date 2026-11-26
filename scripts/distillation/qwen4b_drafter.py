"""Submit and report the Qwen3-4B mask and mask-embedding experiment."""

import argparse
import json
from pathlib import Path
import shlex
from unittest.mock import patch

from scripts import mlspace_launch as launch
from scripts import qwen4b_drafter_data as prepared
from scripts import qwen4b_drafter_eval as evaluation
from scripts import qwen4b_drafter_train as training

from src.experiments.qwen4b_drafter import PHASE_TIMEOUT
PREPARED_TRAIN_TIMEOUT = "24h"
ARCHIVE = "smoltalk-2048"
ARCHIVE_TRAIN = "b0088cb54b3bd646"
ARCHIVE_EVAL = "eval-5b067a4174bd18c8"


def root(dataset):
    design = ARCHIVE_TRAIN if dataset == ARCHIVE else training.signature(dataset)[:16]
    return training.ROOT / design


def marker(phase, name, dataset):
    if phase == "probe":
        return root(dataset) / "prepared.json"
    return root(dataset) / "runs" / "out" / name / (
        "trained.json" if phase == "train" else "evaluated.json"
    )


def names(phase, dataset=training.DEFAULT_DATASET):
    if phase == "probe":
        return ["probe"]
    if phase == "eval":
        if dataset == ARCHIVE:
            return [ARCHIVE_EVAL]
        return ["eval-" + evaluation.signature(dataset)[:16]]
    return [exp.name for exp in training.experiment_configs]


def payload(phase, name, staged, dataset=training.DEFAULT_DATASET):
    body = launch.payload("eval", name, staged, "d.tarasov", "mrsndmn")
    body["job_desc"] = (
        f"qwen4b-drafter {dataset} {phase} {name} {training.signature(dataset)[:16]} "
        "#d.tarasov @mrsndmn"
    )
    timeout = PHASE_TIMEOUT[phase]
    if dataset == prepared.NAME and phase == "train":
        timeout = PREPARED_TRAIN_TIMEOUT
    body["script"] = f"cd {shlex.quote(str(staged))} && " + shlex.join([
        "timeout", "--kill-after=60s", timeout, str(launch.PYTHON),
        "-m", "scripts.qwen4b_drafter", "--phase", phase, "--dataset", dataset,
        "--worker", "--filter", name,
    ])
    return body


def submit(phase, selected, dry, dataset):
    commit = launch.git("rev-parse", "HEAD")
    staged = launch.STAGING_ROOT / commit
    if dry:
        for name in selected:
            print(json.dumps(payload(phase, name, staged, dataset), indent=2))
        return
    if phase == "train" and not marker("probe", "probe", dataset).is_file():
        raise RuntimeError("Finish the GPU probe before submitting training")
    if phase == "eval":
        evaluation.validate_checkpoints(root(dataset) / "runs" / "out", dataset)
    with (
        patch.object(launch, "ARTIFACTS_ROOT", root(dataset)),
        patch.object(launch, "RUN_ROOT", root(dataset) / "runs"),
        launch.submission_lock(),
    ):
        return submit_pending(phase, selected, commit, staged, dataset)


def submit_pending(phase, selected, commit, staged, dataset):
    from mls.manager.job.utils import training_job_api_from_profile

    pending = [name for name in selected if not marker(phase, name, dataset).is_file()]
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
        if marker(phase, name, dataset).is_file():
            continue
        body = payload(phase, name, staged, dataset)
        body["region"] = options["region"]
        identity = launch.normalize(body["job_desc"])
        if identity in active:
            result = {"name": name, "job_name": active[identity], "reused": True}
        else:
            receipt_phase = "prepare" if phase == "probe" else phase
            result = launch.submit_payload(client, body, receipt_phase, name, commit)
        print(json.dumps(result), flush=True)


def worker(phase, name, dataset):
    import torch

    path = marker(phase, name, dataset)
    if path.is_file():
        return
    if phase == "eval":
        evaluation.evaluate(root(dataset) / "runs" / "out", path, dataset)
        return
    if torch.cuda.device_count() != 1:
        raise RuntimeError("This worker requires one visible GPU")
    if phase == "train" and not marker("probe", "probe", dataset).is_file():
        raise RuntimeError("Missing GPU probe")
    target, tokenizer = training.load_target(torch.device("cuda:0"))
    if phase == "train":
        exp = next(exp for exp in training.experiment_configs if exp.name == name)
        training.train(target, tokenizer, exp, path.parent, dataset=dataset)
        return
    reports = [training.train(target, tokenizer, exp, path.parent, probe=True, dataset=dataset)
               for exp in training.experiment_configs]
    if len({row["initial_sha256"] for row in reports}) != 1:
        raise RuntimeError("The four Transformer initializations differ")
    launch.store_json(path, reports)
    print(json.dumps(reports, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(PHASE_TIMEOUT), default="eval")
    parser.add_argument("--dataset", choices=(*training.DATASETS, ARCHIVE))
    parser.add_argument("--filter", help="Exact experiment name")
    mode = parser.add_mutually_exclusive_group()
    for option in ("worker", "dry", "report", "prepare-data"):
        mode.add_argument(f"--{option}", action="store_true")
    args = parser.parse_args()
    dataset = args.dataset or training.DEFAULT_DATASET
    if args.prepare_data:
        if dataset != prepared.NAME or args.filter:
            parser.error(f"--prepare-data requires --dataset {prepared.NAME}")
        path = training.selection_path()
        result = prepared.prepare(path, **training.prepared_settings())
        print(json.dumps(dict(path=str(path), train=len(result["train"]),
                              val=len(result["val"]), rejected=result["rejected"]), indent=2))
        return
    if args.report and args.dataset is None:
        if not any(marker(args.phase, name, dataset).is_file() for name in names(args.phase, dataset)):
            dataset = ARCHIVE
    if dataset == ARCHIVE and not args.report:
        parser.error(f"--dataset {ARCHIVE} is available for --report")
    selected = [name for name in names(args.phase, dataset) if args.filter in (None, name)]
    if not selected:
        parser.error("No matching experiment")
    if args.report:
        for name in selected:
            report = json.loads(marker(args.phase, name, dataset).read_text())
            print(json.dumps(report["summary"] if args.phase == "eval" else report, indent=2))
    elif args.worker:
        if len(selected) != 1:
            parser.error("A worker runs one experiment")
        worker(args.phase, selected[0], dataset)
    else:
        submit(args.phase, selected, args.dry, dataset)


if __name__ == "__main__":
    main()
