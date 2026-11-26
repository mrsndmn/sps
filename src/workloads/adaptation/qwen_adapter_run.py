"""Matched Qwen3 SPS adapter sweep; submit the winner only after all finals exist."""
from orchestration.named.publication import work_path
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex

from orchestration import mlspace_launch as launch
from experiments.qwen_adapter_run import (
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
    arms as configured_arms,
)


SCOPE = os.environ.get('SPS_ADAPTER_SCOPE', 'attention')
if SCOPE not in SWEEPS:
    raise ValueError(f'Unknown adapter scope: {SCOPE}')
_sweep = SWEEPS[SCOPE]
ROOT = COMMON / _sweep.output_name
TARGETS = _sweep.targets
TRAIN_GPUS = _sweep.train_gpus
experiment_configs = configured_arms(SCOPE)


def directory(arm):
    return work_path(ROOT / 'runs/out' / arm.name)


def winner_output():
    return work_path(ROOT / 'evaluations' / 'winner')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def validated_final(arm, commit=None):
    root = directory(arm)
    record = json.loads((root/'trained.json').read_text())
    path = root/'final.pt'
    if (not record.get('commit') or (commit and record['commit'] != commit) or record.get('step') != END or record.get('arm') != arm.name
            or not math.isfinite(record.get('val_nll', math.nan)) or not path.is_file()
            or not path.stat().st_size or digest(path) != record.get('sha256')):
        raise ValueError(f'Invalid finalist: {arm.name}')
    return record


def winner(commit=None):
    scores = []
    cohort = commit
    for arm in experiment_configs:
        record = validated_final(arm, cohort)
        cohort = record["commit"]
        scores.append((record['val_nll'], arm.name, arm))
    return min(scores)[2]


def payload(phase, arm, staged):
    body = launch.payload(phase, arm.name, staged, 'd.tarasov', 'mrsndmn')
    cmd = ['timeout', '8h', str(launch.PYTHON)]
    if phase == 'train':
        cmd += ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={TRAIN_GPUS}']
        body['instance_type'] = f'a100.{TRAIN_GPUS}gpu'
    cmd += ['scripts/adaptation/qwen_adapter_worker.py', phase, arm.name]
    body.update(script=f'cd {shlex.quote(str(staged))} && '+shlex.join(cmd),
                job_desc=f'sps-qwen3-adapters {phase} {arm.name} #d.tarasov @mrsndmn')
    body['env_variables']['SPS_EXPERIMENT_SUITE'] = SUITE
    body['env_variables']['SPS_ADAPTER_SCOPE'] = SCOPE
    body['env_variables']['HF_HUB_OFFLINE'] = '1' if phase == 'train' else '0'
    return body


def submit(phase, dry, name=None):
    commit = launch.git('rev-parse', 'HEAD')
    if not dry and phase == "eval" and not launch.pending(winner_output()):
        return
    arms = experiment_configs if phase == 'train' else [winner()]
    if name is not None:
        arms = [arm for arm in arms if arm.name == name]
        if not arms:
            raise ValueError(f"No selected arm named {name}")
    if dry:
        for arm in arms:
            print(json.dumps(payload(phase, arm, launch.STAGING_ROOT/commit)))
        return
    if phase == "train":
        arms = [arm for arm in arms if launch.pending(directory(arm))]
        if not arms:
            return
    elif winner_output() == ROOT / "evaluations/winner":
        arm = arms[0]
        legacy = directory(arm) / "evaluated.json"
        if legacy.is_file() and json.loads(legacy.read_text()).get("sha256") == validated_final(arm)["sha256"]:
            print(f"Skip existing winner evaluation: {legacy}")
            return
    from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile
    if digest(WHEEL) != WHEEL_SHA or digest(PARENT) != PARENT_SHA:
        raise RuntimeError('Pinned dependency or source digest mismatch')
    staged = launch.stage_repo(Path(launch.git('rev-parse', '--show-toplevel')), commit)
    client, extra = training_job_api_from_profile('default')
    launch.disable_submission_retries(client)
    active = {launch.normalize(job['job_desc']): job['job_name']
              for job in get_in_progress_jobs(client_profile='default')}
    launch.reconcile_for_outputs(client, active, [directory(arm) if phase == "train" else winner_output() for arm in arms])
    for arm in arms:
        body = payload(phase, arm, staged)
        identity = launch.output_identity(body['job_desc'], directory(arm) if phase == "train" else winner_output(), active)
        if identity in active:
            job = active[identity]
            print(job, flush=True)
            continue
        body['region'] = extra['region']
        print(json.dumps(launch.submit_payload(client, body, phase, arm.name, commit, output_dir=directory(arm) if phase == "train" else winner_output())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'eval'))
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--filter', choices=[arm.name for arm in experiment_configs])
    args = parser.parse_args()
    launch.ARTIFACTS_ROOT, launch.RUN_ROOT = ROOT, ROOT/'runs'
    if args.dry:
        return submit(args.phase, True, args.filter)
    with launch.submission_lock():
        submit(args.phase, False, args.filter)
