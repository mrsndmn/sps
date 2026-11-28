"""Publish one shared token corpus and train selected SPS scales on MLSpace."""
import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
from statistics import fmean, median

from experiments.paper_reproduction import PaperSPS, PrepareFineWeb, TokenizePart, TrainSPS, XLProbe
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
                length, run.num_gpus, rank, recipe.block_size, recipe.chunk_size_units or defaults.chunk_size_units,
                recipe.seed, start_offset=offset,
            )
            if len(sampler) < required:
                raise ValueError(f"Insufficient training data for {run.name}, rank {rank}")
    if manifest["tokens"]["val"] < defaults.eval_total_tokens:
        raise ValueError("Insufficient validation data")
    return manifest["tokens"]


def tokenized_parts(corpus):
    ready = [p for p in corpus.parts if p.output.is_dir()]
    missing = [p for p in corpus.parts if p not in ready]
    if missing and (ready or any(RunDirectory(COMMON_ROOT, p.output).reservation.exists()
                                 for p in missing)):
        raise RuntimeError("Await the remaining tokenized parts: " + ", ".join(p.name for p in missing))
    return [(p.output, p.files) for p in ready]


def train_command(run, output):
    command = [str(PYTHON)]
    if run.num_nodes == 1:
        command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={run.num_gpus}"]
    command += ["scripts/train.py", "--recipe", run.recipe.name,
                "--data_root", str(run.corpus.data_root), "--out_dir", str(output), "--wandb_log", "false"]
    if isinstance(run, XLProbe):
        command += ["--init_from", "scratch"]
    return command


def summarize_probe(run, rows):
    batch_tokens = TrainArgs().global_batch_size * run.recipe.block_size
    steps = [r for r in rows if r["event"] == "train"]
    if [r["tokens"] for r in steps] != [batch_tokens * i for i in range(1, run.stop_after_steps + 1)]:
        raise RuntimeError("Incomplete optimizer step sequence")
    evaluations = [r for r in rows if r["event"] in ("validation", "final")]
    if [(r["event"], r["tokens"]) for r in evaluations] != [
            ("validation", 0), ("final", steps[-1]["tokens"])]:
        raise RuntimeError("Missing initial or final validation")
    if not all(math.isfinite(r["loss"]) and 0 < r["step_seconds"] < math.inf for r in steps):
        raise RuntimeError("Invalid training metrics")
    if not all(math.isfinite(r["nll"]) and 0 < r["seconds"] < math.inf for r in evaluations):
        raise RuntimeError("Invalid validation metrics")
    durations = [r["step_seconds"] for r in steps[run.discard_steps:]]
    step = fmean(durations)
    initial_loss = fmean(r["loss"] for r in steps[:run.discard_steps])
    final_loss = fmean(r["loss"] for r in steps[-run.discard_steps:])
    stable = (final_loss <= initial_loss + run.max_loss_increase and
              evaluations[-1]["nll"] <= evaluations[0]["nll"] + run.max_loss_increase)
    validation = evaluations[-1]["seconds"]
    steps_per_eval = run.recipe.eval_interval_tokens / batch_tokens
    baseline = run.baseline_step_seconds + run.baseline_validation_seconds / steps_per_eval
    effective = step + validation / steps_per_eval
    return dict(fingerprint=run.fingerprint, steps=len(steps), median_step_seconds=median(durations),
                mean_step_seconds=step, validation_seconds=validation, stable=stable,
                final_nll=evaluations[-1]["nll"], effective_speedup=baseline / effective,
                faster=step < run.baseline_step_seconds and effective < baseline)


def run_training(run, work):
    primary = run.num_nodes == 1 or os.environ["RANK"] == "0"
    metrics = work / "metrics.jsonl"
    offset = metrics.stat().st_size if primary and metrics.exists() else 0
    subprocess.run(train_command(run, work), check=True)
    if primary:
        if not any(work.glob("*_final.pt")):
            raise RuntimeError("Missing final training checkpoint")
        if isinstance(run, XLProbe):
            with metrics.open("rb") as stream:
                stream.seek(offset)
                report = summarize_probe(run, [json.loads(line) for line in stream])
            (work / "probe.json").write_text(json.dumps(report))
            print(json.dumps(report), flush=True)


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
    corpus = run if isinstance(run, PrepareFineWeb) else run.corpus
    training_runs = [r for r in runs.values() if isinstance(r, TrainSPS) and r.corpus == corpus]
    output = run.output
    if args.worker:
        directory = RunDirectory(COMMON_ROOT, output)
        work = directory.work
        if (os.environ.get("SPS_JOB_OUTPUT") != str(output)
                or os.environ.get("SPS_JOB_WORK") != str(work) or not work.is_dir()):
            parser.error("Worker requires the MLSpace publication directory")
        if training:
            if run.num_nodes == 1 or os.environ["RANK"] == "0":
                check_data(corpus.output, corpus, [run])
            run_training(run, work)
        elif isinstance(run, TokenizePart):
            from data.pilot_data import prepare_memmaps
            # A handover from a serial worker can supply verified complete shards.
            seed = run.output.with_name(run.output.name + "-prefix")
            seed_files = ()
            if seed.is_dir():
                seed_files = json.loads((seed / "manifest.json").read_text())["files"]
            prepare_memmaps(work, run.files, corpus.free_bytes, 1,
                            prefix_dir=seed if seed_files else None, prefix_files=seed_files)
        else:
            from data.pilot_data import prepare_memmaps
            prefix = corpus.prefix if (corpus.prefix / "manifest.json").is_file() else None
            parts = tokenized_parts(corpus)
            if parts and prefix is None:
                pilot = work / "prefix"
                prepare_memmaps(pilot, corpus.prefix_files, corpus.free_bytes, 1)
                prefix = pilot
            prepare_memmaps(work, corpus.files, corpus.free_bytes,
                            max(r.recipe.max_tokens for r in training_runs),
                            prefix_dir=prefix, prefix_files=corpus.prefix_files, parts=parts)
            print(check_data(work, corpus, training_runs))
            if (work / "prefix").is_dir():
                shutil.rmtree(work / "prefix")
        return
    commit = jobs.git("rev-parse", "HEAD")
    staged = jobs.STAGING_ROOT / commit
    body = jobs.payload(run.phase, args.name, staged, "d.tarasov", "mrsndmn",
                        training_gpus=run.num_gpus // run.num_nodes if training else run.num_gpus)
    if training and run.num_nodes > 1:
        per_node = run.num_gpus // run.num_nodes
        body.update(n_workers=run.num_nodes, processes_per_worker=per_node)
        body["env_variables"].update(SPS_MPI_NODES=str(run.num_nodes), SPS_MPI_GPUS_PER_NODE=str(per_node))
    body["job_desc"] = f"sps-{run.family} {run.phase} {args.name} #d.tarasov @mrsndmn"
    body["script"] = shlex.join([str(PYTHON), run.script, args.name, "--worker"])
    settings = run.recipe.arguments() if training else dict(files=run.files, output=str(output))
    if args.dry:
        jobs.print_preview(body, output, staged, run.phase, settings=settings)
        return
    if training and run.probe is not None:
        report = run.probe.output / "probe.json"
        result = json.loads(report.read_text()) if report.is_file() else {}
        if (run.benchmark_workload() != run.probe.benchmark_workload()
                or result.get("fingerprint") != run.probe.fingerprint
                or not all(result.get(k) is True for k in ("faster", "stable"))):
            raise RuntimeError(f"Requires a completed, stable and faster {run.probe.name} probe: {report}")
    if training or (isinstance(run, PrepareFineWeb) and output.is_dir()):
        check_data(corpus.output, corpus, [run] if training else training_runs)
    if isinstance(run, PrepareFineWeb) and not output.is_dir():
        tokenized_parts(corpus)
    with jobs.submission_lock():
        receipt = jobs.SubmissionBatch(commit).submit(body, run.phase, args.name, output, settings=settings)
    print(json.dumps(receipt))
