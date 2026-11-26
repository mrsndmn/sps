"""Publish one shared token corpus and train selected SPS scales on MLSpace."""
import argparse
import json
import math
import os
import shlex
import subprocess

from experiments import paper_reproduction as study
from experiments.environment import PYTHON
from experiments.worker_args import TrainArgs
from orchestration import mlspace_launch as jobs
from orchestration.run_directory import RunDirectory
from experiments.environment import COMMON_ROOT


def check_data(dest):
    from data.pilot_data import checked_manifest
    from training.sampler import FixedRandomChunkDistributedSampler, initial_sampler_offset

    manifest = checked_manifest(dest, study.FILES)
    defaults = TrainArgs()
    for run in study.experiment_configs:
        recipe = run.recipe
        length = manifest["tokens"]["train"] - recipe.block_size
        offset = initial_sampler_offset(length, recipe.seed)
        batch = defaults.global_batch_size
        required = math.ceil(recipe.max_tokens / (batch * recipe.block_size)) * batch // run.num_gpus
        for rank in range(run.num_gpus):
            sampler = FixedRandomChunkDistributedSampler(
                length, run.num_gpus, rank, recipe.block_size, defaults.chunk_size_units,
                recipe.seed, start_offset=offset,
            )
            if len(sampler) < required:
                raise ValueError(f"Insufficient training data for {run.scale}, rank {rank}")
    if manifest["tokens"]["val"] < defaults.eval_total_tokens:
        raise ValueError("Insufficient validation data")
    return manifest["tokens"]


def train_command(run, output):
    return [str(PYTHON), "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={run.num_gpus}", "scripts/train.py",
            "--recipe", run.recipe.name, "--data_root", str(study.ROOT),
            "--out_dir", str(output), "--wandb_log", "false"]


def main():
    runs = {r.scale: r for r in study.experiment_configs}
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=["data", *runs])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry", action="store_true")
    mode.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    run = runs.get(args.name)
    phase = "train" if run else "prepare"
    output = study.ROOT / "out" / run.recipe.name if run else study.CORPUS
    if args.worker:
        directory = RunDirectory(COMMON_ROOT, output)
        work = directory.work
        if (os.environ.get("SPS_JOB_OUTPUT") != str(output)
                or os.environ.get("SPS_JOB_WORK") != str(work) or not work.is_dir()):
            parser.error("Worker requires the MLSpace publication directory")
        if run:
            check_data(study.CORPUS)
            subprocess.run(train_command(run, work), check=True)
            if not any(work.glob("*_final.pt")):
                raise RuntimeError("Missing final training checkpoint")
        else:
            from data.pilot_data import prepare_memmaps
            prefix = study.PREFIX if (study.PREFIX / "manifest.json").is_file() else None
            prepare_memmaps(work, study.FILES, study.PREPARE_FREE_BYTES,
                            max(r.recipe.max_tokens for r in runs.values()),
                            prefix_dir=prefix, prefix_files=study.PREFIX_FILES)
            print(check_data(work))
        return
    commit = jobs.git("rev-parse", "HEAD")
    staged = jobs.STAGING_ROOT / commit
    body = jobs.payload(phase, args.name, staged, "d.tarasov", "mrsndmn",
                        training_gpus=run.num_gpus if run else 1)
    body["job_desc"] = f"sps-{study.FAMILY} {phase} {args.name} #d.tarasov @mrsndmn"
    body["script"] = shlex.join([str(PYTHON), study.SCRIPT, args.name, "--worker"])
    settings = run.recipe.arguments() if run else dict(files=study.FILES, corpus=str(output))
    if args.dry:
        jobs.print_preview(body, output, staged, phase, settings=settings)
        return
    if run or output.is_dir():
        check_data(study.CORPUS)
    with jobs.submission_lock():
        receipt = jobs.SubmissionBatch(commit).submit(body, phase, args.name, output, settings=settings)
    print(json.dumps(receipt))
