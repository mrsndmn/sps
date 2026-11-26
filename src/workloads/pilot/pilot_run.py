"""Job-side preparation/probes, bounded training and checkpoint-scoped evaluation."""
import argparse
import json
import os
from pathlib import Path
import statistics
from orchestration.pilot_launch import destination, prepare_report
import math
import subprocess
import time

from orchestration.experiment_suite import (RUN_ROOT, SANITY_ROOT, MAX_HOURS, TASKS, collect_experiments,
                        train_command, eval_command,
                        COMMON_ROOT, RUNTIME_SAFETY_FACTOR)


def run_logged(command, log, *, timeout):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as handle:
        # A separate process group lets timeout terminate torchrun's workers too.
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            import signal
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code:
        raise RuntimeError(f'Command failed ({code}); inspect {log}')


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def estimate(exp, probe_root=None):
    probe_root = SANITY_ROOT if probe_root is None else probe_root
    training = exp.training
    tokens_per_step = training.global_batch_size * training.block_size
    records = [json.loads(line) for line in
        (probe_root / 'out' / exp.name / 'metrics.jsonl').read_text().splitlines()]
    steps = [r['step_seconds'] for r in records if r['event'] == 'train'][-8:]
    if len(steps) < 8 or any(v <= 0 for v in steps):
        raise RuntimeError('Probe did not produce eight stable step measurements')
    seconds = statistics.median(steps)
    reference = None
    if exp.reference_name is not None:
        reference_path = COMMON_ROOT / 'runs' / 'out' / exp.reference_name
        prior = json.loads((reference_path / 'trained.json').read_text())
        prior_rows = [json.loads(line) for line in
                      (reference_path / 'metrics.jsonl').read_text().splitlines()]
        final = next(r for r in reversed(prior_rows) if r['event'] == 'final')
        # Include all observed validation/checkpoint overhead from the completed
        # reference run; never replace a slower fresh probe by a faster history.
        prior_step = prior['seconds'] / (final['tokens'] / tokens_per_step)
        seconds = max(seconds, prior_step)
        reference = dict(name=exp.reference_name, seconds_per_step=prior_step)
    projected = seconds * (training.max_tokens / tokens_per_step) * RUNTIME_SAFETY_FACTOR + 3600
    return dict(step_seconds=seconds, tokens_per_second=tokens_per_step / seconds,
                projected_training_hours=projected / 3600,
                token_budget=training.max_tokens, max_hours=MAX_HOURS,
                safety_factor=RUNTIME_SAFETY_FACTOR, reference=reference)


def validate_training_coverage(manifest):
    from training.sampler import FixedRandomChunkDistributedSampler, initial_sampler_offset
    coverage = {}
    for exp in collect_experiments():
        training = exp.training
        dataset_len = manifest['tokens']['train'] - training.block_size
        offset = initial_sampler_offset(dataset_len, training.seed)
        tokens_per_step = training.global_batch_size * training.block_size
        required = math.ceil(training.max_tokens / tokens_per_step) * training.global_batch_size // exp.num_gpus
        capacities = []
        for rank in range(exp.num_gpus):
            sampler = FixedRandomChunkDistributedSampler(dataset_len, exp.num_gpus,
                rank, training.block_size, chunk_size_units=training.chunk_size_units,
                seed=training.seed, start_offset=offset)
            if len(sampler) < required:
                raise RuntimeError(f'Insufficient unique training data for rank {rank}: '
                                   f'{len(sampler)} sequence starts, need {required}')
            capacities.append(len(sampler))
        coverage[exp.name] = dict(required_per_rank=required, available_per_rank=capacities)
    return coverage


def prepare(work):
    import torch
    required_gpus = max(exp.num_gpus for exp in collect_experiments())
    if torch.cuda.device_count() != required_gpus:
        raise RuntimeError(f'Preparation/probe job requires {required_gpus} visible GPUs')
    from orchestration.pilot_headers import prepare as prepare_headers
    prepare_headers()
    from experiments.environment import PYTHON
    run_logged([str(PYTHON), '-m', 'pytest', '-q',
                'src/modeling/tests/attention/triton/test_sps_attention.py',
                '-k', 'sliding_matches'], work / 'mask-tests.log', timeout=1800)
    from data.pilot_data import prepare as prepare_data
    manifest = prepare_data()
    coverage = validate_training_coverage(manifest)
    work.mkdir(parents=True, exist_ok=True)
    data_link = work / 'data'
    if not data_link.exists():
        data_link.symlink_to(RUN_ROOT / 'data', target_is_directory=True)
    estimates = {}
    for exp in collect_experiments():
        metrics = work / 'out' / exp.name / 'metrics.jsonl'
        # Probe always starts from scratch; do not mix timings from prior attempts.
        if metrics.exists():
            metrics.unlink()
        run_logged(train_command(exp, probe=True, data_root=work), work / f'{exp.name}.log', timeout=3600)
        estimates[exp.name] = estimate(exp, work)
        print(exp.name, estimates[exp.name], flush=True)
    if max(e['projected_training_hours'] for e in estimates.values()) > MAX_HOURS:
        raise RuntimeError('Measured training exceeds time budget; reduce matched budget in code')
    write_json(work / 'prepared.json', dict(data=manifest, estimates=estimates, coverage=coverage))


def train(exp, dest):
    prepared = json.loads(prepare_report().read_text())
    prediction = prepared['estimates'][exp.name]
    if prediction['token_budget'] != exp.training.max_tokens or prediction['projected_training_hours'] > MAX_HOURS:
        raise RuntimeError('Run preparation again for the current experiment budget')
    write_json(dest / 'run-config.json', dict(settings=exp.config(RUN_ROOT)))
    started = time.time()
    run_logged(train_command(exp, dest=dest), dest / 'train.log', timeout=MAX_HOURS * 3600)
    finals = list(dest.glob('*_final.pt'))
    if not finals:
        raise RuntimeError('Training exited without final checkpoint')
    write_json(dest / 'trained.json', dict(checkpoint=str(exp.output_dir / finals[-1].name),
               seconds=time.time() - started, estimate=prediction))


def evaluate(exp, dest):
    trained = json.loads((exp.output_dir / 'trained.json').read_text())
    checkpoint = Path(trained['checkpoint'])
    for task in TASKS:
        result = dest / checkpoint.stem / f'{task}.json'
        run_logged(eval_command(exp, task, result=result), dest / f'eval-{task}.log', timeout=6 * 3600)
        if not result.is_file():
            raise RuntimeError(f'Missing task result {result}')
    write_json(dest / 'evaluated.json', dict(tasks=TASKS, checkpoint=str(checkpoint)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'train', 'eval'))
    parser.add_argument('--filter')
    parser.add_argument("--reserved", action="store_true")
    args = parser.parse_args()
    exp = None
    if args.phase != 'prepare':
        exp, = [run for run in collect_experiments(args.filter) if run.name == args.filter]
    directory = destination(args.phase, exp)
    if not args.reserved and not directory.reserve():
        return
    if not directory.reservation.is_dir():
        raise RuntimeError(f'Missing submission reservation {directory.reservation}')
    try:
        if args.phase == 'prepare':
            prepare(directory.work)
        else:
            (train if args.phase == 'train' else evaluate)(exp, directory.work)
        directory.finish()
    except Exception:
        if not args.reserved and directory.reservation.is_dir():
            directory.retire()
        raise
