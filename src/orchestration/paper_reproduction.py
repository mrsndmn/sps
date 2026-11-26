"""Publish one shared token corpus and train selected SPS scales on MLSpace."""
import argparse
import json
import math
import os
import shlex
import subprocess

from experiments.paper_reproduction import PaperSPS, TrainSPS
from experiments.environment import COMMON_ROOT, PYTHON
from experiments.worker_args import TrainArgs
from orchestration import mlspace_launch as jobs
from orchestration.run_directory import RunDirectory


def check_data(dest, corpus, training_runs):
    from data.pilot_data import checked_manifest
    from training.sampler import FixedRandomChunkDistributedSampler, initial_sampler_offset

    manifest = checked_manifest(dest, corpus.files)
    defaults = TrainArgs()
    for run in training_runs:
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
                raise ValueError(f"Insufficient training data for {run.name}, rank {rank}")
    if manifest["tokens"]["val"] < defaults.eval_total_tokens:
        raise ValueError("Insufficient validation data")
    return manifest["tokens"]


def train_command(run, output):
    return [str(PYTHON), "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={run.num_gpus}", "scripts/train.py",
            "--recipe", run.recipe.name, "--data_root", str(run.corpus.data_root),
            "--out_dir", str(output), "--wandb_log", "false"]


def main():
    runs = {r.name: r for r in PaperSPS.experiment_configs()}
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=runs)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry", action="store_true")
    mode.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    run = runs[args.name]
    training = isinstance(run, TrainSPS)
    corpus = run.corpus if training else run
    training_runs = [r for r in runs.values() if isinstance(r, TrainSPS) and r.corpus == corpus]
    output = run.output
    if args.worker:
        directory = RunDirectory(COMMON_ROOT, output)
        work = directory.work
        if (os.environ.get("SPS_JOB_OUTPUT") != str(output)
                or os.environ.get("SPS_JOB_WORK") != str(work) or not work.is_dir()):
            parser.error("Worker requires the MLSpace publication directory")
        if training:
            check_data(corpus.output, corpus, training_runs)
            subprocess.run(train_command(run, work), check=True)
            if not any(work.glob("*_final.pt")):
                raise RuntimeError("Missing final training checkpoint")
        else:
            from data.pilot_data import prepare_memmaps
            prefix = corpus.prefix if (corpus.prefix / "manifest.json").is_file() else None
            prepare_memmaps(work, corpus.files, corpus.free_bytes,
                            max(r.recipe.max_tokens for r in training_runs),
                            prefix_dir=prefix, prefix_files=corpus.prefix_files)
            print(check_data(work, corpus, training_runs))
        return
    commit = jobs.git("rev-parse", "HEAD")
    staged = jobs.STAGING_ROOT / commit
    body = jobs.payload(run.phase, args.name, staged, "d.tarasov", "mrsndmn",
                        training_gpus=run.num_gpus)
    body["job_desc"] = f"sps-{run.family} {run.phase} {args.name} #d.tarasov @mrsndmn"
    body["script"] = shlex.join([str(PYTHON), run.script, args.name, "--worker"])
    settings = run.recipe.arguments() if training else dict(files=corpus.files, corpus=str(output))
    if args.dry:
        jobs.print_preview(body, output, staged, run.phase, settings=settings)
        return
    if training or output.is_dir():
        check_data(corpus.output, corpus, training_runs)
    with jobs.submission_lock():
        receipt = jobs.SubmissionBatch(commit).submit(body, run.phase, args.name, output, settings=settings)
    print(json.dumps(receipt))
