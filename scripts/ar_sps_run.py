"""Bounded workers for the isolated AR-to-SPS suite."""
import argparse
import json
import math
import os
import subprocess
import time

import experiments as e
from scripts import ar_sps_config as c
from scripts.mlspace_launch import store_json


def hours(phase):
    return {'prepare': c.PREPARE_HOURS, 'train': c.TRAIN_HOURS, 'eval': c.EVAL_HOURS}[phase]


def reserve(phase, name):
    """Called under the existing submission lock; failed attempts remain charged.

    Conservative reservation accounting prevents retries from spending the cap
    twice. Queue time is not charged; worker startup has an explicit reserve.
    """
    path = e.ARTIFACTS_ROOT/'budget.json'
    ledger = json.loads(path.read_text()) if path.exists() else {'attempts': []}
    cost = hours(phase)*(1 if phase == 'eval' else 4)
    used = sum(row['gpu_hours'] for row in ledger['attempts'])
    if used+cost > c.GPU_HOURS:
        raise RuntimeError(f'Aggregate GPU-hour reservation cap exceeded: {used}+{cost}>{c.GPU_HOURS}')
    ledger['attempts'].append(dict(phase=phase, name=name, gpu_hours=cost, time=time.time()))
    store_json(path, ledger)


def calibrated_plan(probes):
    rates = []
    for name, probe in probes.items():
        if probe['full'] <= 0 or not math.isfinite(probe['full']):
            raise RuntimeError(f'Invalid full-stage throughput for {name}')
        if 'staged' in name:
            if probe['embedding'] <= 0 or not math.isfinite(probe['embedding']):
                raise RuntimeError('Missing frozen-stage throughput')
            rates.append(c.STAGE_FRACTION*probe['embedding'] + (1-c.STAGE_FRACTION)*probe['full'])
        else:
            rates.append(probe['full'])
    available = c.TRAIN_HOURS*3600-c.STARTUP_SECONDS
    steps = min(c.MAX_STEPS, int(available/(max(rates)*c.SAFETY)))
    if steps < c.MIN_STEPS:
        raise RuntimeError('No meaningful common training budget fits the cap')
    return dict(steps=steps, stage_steps=max(1, int(steps*c.STAGE_FRACTION)),
                text_tokens=steps*c.GLOBAL_BATCH*c.CONTEXT, probes=probes,
                design_signature=e.design_signature())


def internal(phase, name):
    if phase == 'prepare':
        from scripts.ar_sps_data import prepare
        prepare()
        probes = {}
        for exp in e.collect_experiments():
            if not exp.trainable:
                continue
            subprocess.run(e.train_command(exp, probe=True), check=True)
            probes[exp.name] = json.loads((e.SANITY_ROOT/'out'/exp.name/'probe.json').read_text())
        # Native AR parity is a hard preparation gate, before marker publication.
        subprocess.run([str(e.PYTHON), '-m', 'scripts.ar_sps_evaluate', '--audit'], check=True)
        store_json(e.ARTIFACTS_ROOT/'prepared.json', calibrated_plan(probes))
    else:
        matches = [exp for exp in e.collect_experiments() if exp.name == name]
        if len(matches) != 1:
            raise ValueError('Unknown experiment')
        exp = matches[0]
        if phase == 'train':
            subprocess.run(e.train_command(exp), check=True)
        else:
            subprocess.run([str(e.PYTHON), '-m', 'scripts.ar_sps_evaluate', '--filter', name], check=True)
        store_json(e.output_dir(exp)/('trained.json' if phase == 'train' else 'evaluated.json'),
                   dict(design_signature=e.design_signature()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'train', 'eval'))
    parser.add_argument('--filter')
    parser.add_argument('--internal', action='store_true')
    args = parser.parse_args()
    if e.SUITE != 'ar-sps':
        raise RuntimeError('Select SPS_EXPERIMENT_SUITE=ar-sps')
    if args.internal:
        return internal(args.phase, args.filter)
    # A process group ensures a timeout kills torchrun and every rank, not only
    # the wrapper. Deadline includes downloads, validation and checkpoint writes.
    command = [str(e.PYTHON), '-m', 'scripts.ar_sps_run', args.phase, '--internal']
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
