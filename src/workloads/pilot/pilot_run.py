"""Job-side preparation/probes, bounded training and checkpoint-scoped evaluation."""
import argparse
import json
import os
from pathlib import Path
import statistics
import math
import subprocess
import time

from orchestration.experiment_suite import (ARTIFACTS_ROOT, RUN_ROOT, SANITY_ROOT, MAX_HOURS, TASKS, collect_experiments,
                        train_command, eval_command, output_dir, design_signature,
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


def estimate(exp):
    training = exp.training
    tokens_per_step = training.global_batch_size * training.block_size
    records = [json.loads(line) for line in
        (SANITY_ROOT / 'out' / exp.name / 'metrics.jsonl').read_text().splitlines()]
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


def prepare():
    import torch
    required_gpus = max(exp.num_gpus for exp in collect_experiments())
    if torch.cuda.device_count() != required_gpus:
        raise RuntimeError(f'Preparation/probe job requires {required_gpus} visible GPUs')
    from orchestration.pilot_headers import prepare as prepare_headers
    prepare_headers()
    from experiments.environment import PYTHON
    run_logged([str(PYTHON), '-m', 'pytest', '-q',
                'src/modeling/tests/attention/triton/test_sps_attention.py',
                '-k', 'sliding_matches'], SANITY_ROOT / 'mask-tests.log', timeout=1800)
    from data.pilot_data import prepare as prepare_data
    manifest = prepare_data()
    coverage = validate_training_coverage(manifest)
    SANITY_ROOT.mkdir(parents=True, exist_ok=True)
    data_link = SANITY_ROOT / 'data'
    if not data_link.exists():
        data_link.symlink_to(RUN_ROOT / 'data', target_is_directory=True)
    estimates = {}
    for exp in collect_experiments():
        metrics = SANITY_ROOT / 'out' / exp.name / 'metrics.jsonl'
        # Probe always starts from scratch; do not mix timings from prior attempts.
        if metrics.exists():
            metrics.unlink()
        run_logged(train_command(exp, probe=True), SANITY_ROOT / f'{exp.name}.log', timeout=3600)
        estimates[exp.name] = estimate(exp)
        print(exp.name, estimates[exp.name], flush=True)
    if max(e['projected_training_hours'] for e in estimates.values()) > MAX_HOURS:
        raise RuntimeError('Measured training exceeds time budget; reduce matched budget in code')
    write_json(ARTIFACTS_ROOT / 'prepared.json', dict(data=manifest, estimates=estimates, coverage=coverage,
                                                   design_signature=design_signature()))


def train(exp):
    prepared = json.loads((ARTIFACTS_ROOT / 'prepared.json').read_text())
    if prepared['design_signature'] != design_signature():
        raise RuntimeError('Stale preparation; rerun probes for the current design')
    prediction = prepared['estimates'][exp.name]
    if prediction['token_budget'] != exp.training.max_tokens or prediction['projected_training_hours'] > MAX_HOURS:
        raise RuntimeError('Run preparation again for the current experiment budget')
    dest = output_dir(exp)
    identity = dest / 'run-config.json'
    if identity.exists():
        if json.loads(identity.read_text())['design_signature'] != design_signature():
            raise RuntimeError('Partial checkpoints belong to a different design; use a new run name')
    else:
        if list(dest.glob('*.pt')):
            raise RuntimeError('Cannot resume checkpoints without an experiment identity')
        write_json(identity, dict(design_signature=design_signature(), command=train_command(exp),
                                  settings=exp.config(RUN_ROOT)))
    if (dest / 'trained.json').exists():
        return
    started = time.time()
    run_logged(train_command(exp), dest / 'train.log', timeout=MAX_HOURS * 3600)
    finals = list(dest.glob('*_final.pt'))
    if not finals:
        raise RuntimeError('Training exited without final checkpoint')
    write_json(dest / 'trained.json', dict(checkpoint=str(finals[-1]), design_signature=design_signature(),
               seconds=time.time() - started, estimate=prediction))


def evaluate(exp):
    dest = output_dir(exp)
    if not (dest / 'trained.json').exists():
        raise RuntimeError(f'No completed training at {dest}')
    if json.loads((dest / 'trained.json').read_text())['design_signature'] != design_signature():
        raise RuntimeError('Training marker belongs to a different experiment design')
    for task in TASKS:
        marker = dest / f'evaluated-{task}.json'
        if marker.exists():
            continue
        run_logged(eval_command(exp, task), dest / f'eval-{task}.log', timeout=6 * 3600)
        write_json(marker, dict(task=task, checkpoint=json.loads(
            (dest / 'trained.json').read_text())['checkpoint']))
    write_json(dest / 'evaluated.json', dict(tasks=TASKS, design_signature=design_signature()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'train', 'eval'))
    parser.add_argument('--filter')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    else:
        matches = collect_experiments(args.filter)
        if len(matches) != 1:
            raise ValueError('Specify exactly one experiment')
        (train if args.phase == 'train' else evaluate)(matches[0])
