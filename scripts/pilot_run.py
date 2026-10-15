"""Job-side preparation/probes, bounded training and checkpoint-scoped evaluation."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from experiments import (ARTIFACTS_ROOT, RUN_ROOT, SANITY_ROOT, TRAIN_TOKENS,
                        TOKENS_PER_STEP, MAX_HOURS, TASKS, collect_experiments,
                        train_command, eval_command, output_dir, design_signature)


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
    records = [json.loads(line) for line in
        (SANITY_ROOT / 'out' / exp.name / 'metrics.jsonl').read_text().splitlines()]
    steps = [r['step_seconds'] for r in records if r['event'] == 'train'][-8:]
    if len(steps) < 8 or any(v <= 0 for v in steps):
        raise RuntimeError('Probe did not produce eight stable step measurements')
    seconds = statistics.median(steps)
    # Safety factor includes data/communication variation, plus one hour for
    # validation, checkpoint I/O and startup. Downstream evaluation is separate.
    projected = seconds * (TRAIN_TOKENS / TOKENS_PER_STEP) * 1.5 + 3600
    return dict(step_seconds=seconds, tokens_per_second=TOKENS_PER_STEP / seconds,
                projected_training_hours=projected / 3600,
                token_budget=TRAIN_TOKENS, max_hours=MAX_HOURS)


def prepare():
    import torch
    if torch.cuda.device_count() != 4:
        raise RuntimeError('Preparation/probe job requires exactly four visible GPUs')
    from scripts.pilot_headers import prepare as prepare_headers
    prepare_headers()
    from experiments import PYTHON
    run_logged([str(PYTHON), '-m', 'pytest', '-q',
                'src/modeling/tests/attention/triton/test_sps_attention.py',
                '-k', 'sliding_matches'], SANITY_ROOT / 'mask-tests.log', timeout=1800)
    from scripts.pilot_data import prepare as prepare_data
    manifest = prepare_data()
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
    write_json(ARTIFACTS_ROOT / 'prepared.json', dict(data=manifest, estimates=estimates,
                                                   design_signature=design_signature()))


def train(exp):
    prepared = json.loads((ARTIFACTS_ROOT / 'prepared.json').read_text())
    if prepared['design_signature'] != design_signature():
        raise RuntimeError('Stale preparation; rerun probes for the current design')
    prediction = prepared['estimates'][exp.name]
    if prediction['token_budget'] != TRAIN_TOKENS or prediction['projected_training_hours'] > MAX_HOURS:
        raise RuntimeError('Run preparation again for the current experiment budget')
    dest = output_dir(exp)
    identity = dest / 'run-config.json'
    if identity.exists():
        if json.loads(identity.read_text())['design_signature'] != design_signature():
            raise RuntimeError('Partial checkpoints belong to a different design; use a new run name')
    else:
        if list(dest.glob('*.pt')):
            raise RuntimeError('Cannot resume checkpoints without an experiment identity')
        write_json(identity, dict(design_signature=design_signature(), command=train_command(exp)))
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


if __name__ == '__main__':
    main()
