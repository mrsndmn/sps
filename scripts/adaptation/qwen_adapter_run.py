"""Matched Qwen3 SPS adapter sweep; submit the winner only after all finals exist."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex

from scripts import mlspace_launch as launch
from src.experiments.qwen_adapter_run import (
    SUITE,
    SOURCE,
    PARENT,
    PARENT_SHA,
    SIGNATURE,
    WHEEL,
    WHEEL_SHA,
    DATA_SHA,
    START,
    END,
    RANK,
    ALPHA,
    PRED_LR,
    Arm,
    COMMON,
    SWEEPS,
)


SCOPE = os.environ.get('SPS_ADAPTER_SCOPE', 'attention')
if SCOPE not in SWEEPS:
    raise ValueError(f'Unknown adapter scope: {SCOPE}')
_sweep = SWEEPS[SCOPE]
ROOT = COMMON / _sweep.output_name
TARGETS = _sweep.targets
TRAIN_GPUS = _sweep.train_gpus
experiment_configs = list(_sweep.arms)


def directory(arm):
    return ROOT / 'runs/out' / arm.name


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def validated_final(arm, commit):
    root = directory(arm)
    record = json.loads((root/'trained.json').read_text())
    path = root/'final.pt'
    if (record.get('commit') != commit or record.get('step') != END or record.get('arm') != arm.name
            or not math.isfinite(record.get('val_nll', math.nan)) or not path.is_file()
            or not path.stat().st_size or digest(path) != record.get('sha256')):
        raise ValueError(f'Invalid finalist: {arm.name}')
    return record


def winner(commit):
    scores = []
    for arm in experiment_configs:
        record = validated_final(arm, commit)
        scores.append((record['val_nll'], arm.name, arm))
    return min(scores)[2]


def payload(phase, arm, staged):
    body = launch.payload(phase, arm.name, staged, 'd.tarasov', 'mrsndmn')
    cmd = ['timeout', '8h', str(launch.PYTHON), '-m']
    if phase == 'train':
        cmd += ['torch.distributed.run', '--standalone', f'--nproc_per_node={TRAIN_GPUS}', '-m']
        body['instance_type'] = f'a100.{TRAIN_GPUS}gpu'
    cmd += ['scripts.qwen_adapter_worker', phase, arm.name]
    body.update(script=f'cd {shlex.quote(str(staged))} && '+shlex.join(cmd),
                job_desc=f'sps-qwen3-adapters {phase} {arm.name} #d.tarasov @mrsndmn')
    body['env_variables']['SPS_EXPERIMENT_SUITE'] = SUITE
    body['env_variables']['SPS_ADAPTER_SCOPE'] = SCOPE
    body['env_variables']['HF_HUB_OFFLINE'] = '1' if phase == 'train' else '0'
    return body


def submit(phase, dry):
    commit = launch.git('rev-parse', 'HEAD')
    arms = experiment_configs if phase == 'train' else [winner(commit)]
    if dry:
        for arm in arms:
            print(json.dumps(payload(phase, arm, launch.STAGING_ROOT/commit)))
        return
    from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile
    if digest(WHEEL) != WHEEL_SHA or digest(PARENT) != PARENT_SHA:
        raise RuntimeError('Pinned dependency or source digest mismatch')
    staged = launch.stage_repo(Path(launch.git('rev-parse', '--show-toplevel')), commit)
    client, extra = training_job_api_from_profile('default')
    launch.disable_submission_retries(client)
    active = {launch.normalize(job['job_desc']): job['job_name']
              for job in get_in_progress_jobs(client_profile='default')}
    launch.reconcile_receipts(client, active)
    launch.reconcile_intents(active)
    for arm in arms:
        marker = directory(arm)/('trained.json' if phase == 'train' else 'evaluated.json')
        if marker.exists():
            if json.loads(marker.read_text())['commit'] != commit:
                raise RuntimeError(f'Stale marker: {marker}')
            if phase == 'train':
                try:
                    validated_final(arm, commit)
                except ValueError:
                    pass
                else:
                    continue
            elif json.loads(marker.read_text()).get('sha256') == validated_final(arm, commit)['sha256']:
                continue
        body = payload(phase, arm, staged)
        identity = launch.normalize(body['job_desc'])
        if identity in active:
            job = active[identity]
            if json.loads((ROOT/'submissions'/f'{job}.json').read_text())['commit'] != commit:
                raise RuntimeError('Active job uses another commit')
            print(job, flush=True)
            continue
        body['region'] = extra['region']
        print(json.dumps(launch.submit_payload(client, body, phase, arm.name, commit)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'eval'))
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()
    launch.ARTIFACTS_ROOT, launch.RUN_ROOT = ROOT, ROOT/'runs'
    if args.dry:
        return submit(args.phase, True)
    with launch.submission_lock():
        submit(args.phase, False)


if __name__ == '__main__':
    main()
