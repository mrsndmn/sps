"""Matched Qwen3 SPS adapter sweep; submit the winner only after all finals exist."""
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shlex

from scripts import mlspace_launch as launch


SUITE = 'ar-sps-qwen3-1.7b'
COMMON = launch.STAGING_ROOT.parent
SOURCE = COMMON / SUITE / 'runs'
PARENT = SOURCE / 'out/qwen3_staged_sps_ce/stage_endpoint.pt'
PARENT_SHA = '6ee490f323051d482770bfe0cabc938060caf95f672c39d6d73e117dbf77c1ae'
SIGNATURE = '7dad73e5302bf23ef6bf64cb2b1ce9d7aa6633e7213b2db92fa79bb43d14b1c3'
ROOT = COMMON / 'qwen3-base-adapters'
WHEEL = COMMON / 'toolchain/peft-0.17.1-py3-none-any.whl'
WHEEL_SHA = '3d129d64def3d74779c32a080d2567e5f7b674e77d546e3585138216d903f99e'
DATA_SHA = dict(train='5b49b463f888c67072554d57972607a98dd10997d99d8ff09fa9586f59152796',
                val='51cbd70db36d60abbc0d7f7195743bee535be5edaf377e92b170323cff2da4e0',
                test='75322f02d3d55167129d4a4e8058321ae0f1bcb9982a82c5d9b36c9686b8ec58')
START, END = 819, 4096
RANK, ALPHA = 16, 32
TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
PRED_LR = 1e-5


@dataclass(frozen=True)
class Arm:
    name: str
    dora: bool
    rs: bool
    lr: float


experiment_configs = [Arm(f'{kind}-{scale}-{rate}', kind == 'dora', scale == 'rs', float(rate))
                      for rate in ('3e-5', '1e-4') for kind in ('lora', 'dora')
                      for scale in ('standard', 'rs')]


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
        cmd += ['torch.distributed.run', '--standalone', '--nproc_per_node=4', '-m']
    cmd += ['scripts.qwen_adapter_worker', phase, arm.name]
    body.update(script=f'cd {shlex.quote(str(staged))} && '+shlex.join(cmd),
                job_desc=f'sps-qwen3-adapters {phase} {arm.name} #d.tarasov @mrsndmn')
    body['env_variables']['SPS_EXPERIMENT_SUITE'] = SUITE
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
