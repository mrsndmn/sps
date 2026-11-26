"""Bounded instruct preparation/training/evaluation workers and GPU reservations."""
import argparse
import json
import math
import os
import subprocess
import time

import experiments as e
from scripts import instruct_config as c
from scripts.mlspace_launch import store_json


def hours(phase):
    return {'prepare': c.PREPARE_HOURS, 'train': c.TRAIN_HOURS, 'eval': c.EVAL_HOURS}[phase]


def reserve(phase, name):
    """Called under submission lock; failed attempts retain their reservations."""
    path = e.ARTIFACTS_ROOT/'budget.json'
    ledger = json.loads(path.read_text()) if path.exists() else {'attempts': []}
    cost = hours(phase) * (1 if phase == 'eval' else 4)
    used = sum(row['gpu_hours'] for row in ledger['attempts'])
    if used + cost > c.GPU_HOURS:
        raise RuntimeError(f'Aggregate GPU-hour reservation cap exceeded: {used}+{cost}>{c.GPU_HOURS}')
    ledger['attempts'].append(dict(phase=phase, name=name, gpu_hours=cost, time=time.time()))
    store_json(path, ledger)


def calibrated_plan(probes, training_rows):
    rates = []
    expected = {exp.name: exp for exp in e.collect_experiments() if exp.trainable}
    if set(probes) != set(expected):
        raise RuntimeError('Missing training-arm probes')
    for name, probe in probes.items():
        if probe.get('design_signature') != e.design_signature():
            raise RuntimeError('Stale throughput probe')
        if not math.isfinite(probe['full']) or probe['full'] <= 0:
            raise RuntimeError(f'Invalid full-stage throughput for {name}')
        if expected[name].recipe == 'staged_sps':
            if not math.isfinite(probe['embedding']) or probe['embedding'] <= 0:
                raise RuntimeError(f'Missing frozen-stage throughput for {name}')
            rates.append(c.STAGE_FRACTION * probe['embedding'] + (1-c.STAGE_FRACTION) * probe['full'])
        else:
            rates.append(probe['full'])
    available = c.TRAIN_HOURS * 3600 - c.STARTUP_SECONDS
    steps = min(c.MAX_STEPS, len(training_rows)//c.GLOBAL_BATCH,
                int(available/(max(rates)*c.SAFETY)))
    if steps < c.MIN_STEPS:
        raise RuntimeError('Insufficient common data/time budget for meaningful instruct training')
    selected = training_rows[:steps*c.GLOBAL_BATCH]
    mode_counts = {mode: sum(row['mode'] == mode for row in selected) for mode in ('thinking', 'nonthinking')}
    if not all(mode_counts.values()):
        raise RuntimeError('Calibrated training prefix does not contain both modes')
    stage_steps = max(1, int(steps*c.STAGE_FRACTION))
    if stage_steps >= steps:
        raise RuntimeError('No full training stage fits')
    return dict(steps=steps, stage_steps=stage_steps, examples=len(selected),
                text_tokens=sum(len(row['input_ids'])-1 for row in selected),
                target_tokens=sum(sum(row['loss_mask'][1:]) for row in selected),
                mode_counts=mode_counts,
                target_tokens_by_mode={mode:sum(sum(row['loss_mask'][1:]) for row in selected if row['mode']==mode)
                                       for mode in mode_counts},
                probes=probes, design_signature=e.design_signature())


def internal(phase, name):
    if phase == 'prepare':
        subprocess.run([str(e.PYTHON), '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                        '-m', 'scripts.instruct_data'], check=True)
        probes = {}
        for exp in e.collect_experiments():
            if exp.trainable:
                subprocess.run(e.train_command(exp, probe=True), check=True)
                probes[exp.name] = json.loads((e.SANITY_ROOT/'out'/exp.name/'probe.json').read_text())
        # This is a numerical preparation gate, not a benchmark evaluation job.
        subprocess.run([str(e.PYTHON), '-m', 'scripts.instruct_evaluate', '--audit'], check=True)
        from scripts.instruct_data import rows
        plan = calibrated_plan(probes, rows('train'))
        store_json(e.ARTIFACTS_ROOT/'prepared.json', plan)
        return
    matches = [exp for exp in e.collect_experiments() if exp.name == name]
    if len(matches) != 1:
        raise ValueError('Select exactly one experiment')
    exp = matches[0]
    prepared = json.loads((e.ARTIFACTS_ROOT/'prepared.json').read_text())
    if prepared['design_signature'] != e.design_signature():
        raise RuntimeError('Stale preparation plan')
    if phase == 'train':
        if not exp.trainable:
            raise ValueError('Source model is not a training arm')
        subprocess.run(e.train_command(exp), check=True)
        from scripts.instruct_common import validate_final
        validate_final(e.output_dir(exp)/'final.pt', prepared)
        store_json(e.output_dir(exp)/'trained.json', dict(design_signature=e.design_signature(),
                                                        plan=prepared, checkpoint='final.pt'))
    else:
        subprocess.run([str(e.PYTHON), '-m', 'scripts.instruct_evaluate', '--filter', name], check=True)
        # Preserve evaluator-owned evaluated.json, including task coverage and audit.
        marker = json.loads((e.output_dir(exp)/'evaluated.json').read_text())
        if marker['design_signature'] != e.design_signature():
            raise RuntimeError('Evaluator did not publish a matching completion marker')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'train', 'eval'))
    parser.add_argument('--filter')
    parser.add_argument('--internal', action='store_true')
    args = parser.parse_args()
    if e.SUITE != 'ar-sps-qwen3-1.7b-instruct':
        raise RuntimeError('Select SPS_EXPERIMENT_SUITE=ar-sps-qwen3-1.7b-instruct')
    if args.internal:
        return internal(args.phase, args.filter)
    command = [str(e.PYTHON), '-m', 'scripts.instruct_run', args.phase, '--internal']
    if args.filter:
        command += ['--filter', args.filter]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        result = process.wait(timeout=hours(args.phase)*3600-c.STARTUP_SECONDS)
        if result:
            raise RuntimeError(f'{args.phase} worker exited {result}')
    except BaseException:
        import signal
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise


if __name__ == '__main__':
    main()
