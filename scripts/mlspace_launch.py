"""Shared, side-effect-free planning and committed-code MLSpace submission."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import uuid

from experiments import (ARTIFACTS_ROOT, RUN_ROOT, STAGING_ROOT, HF_HOME,
                        PYTHON, BASE_IMAGE, INSTANCE_TYPES_BY_NUM_GPUS,
                        collect_experiments, output_dir, design_signature)
from scripts.pilot_headers import HEADER_ROOT


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def stage_repo(repo, commit):
    if git('-C', str(repo), 'diff', '--name-only', 'HEAD'):
        raise RuntimeError('Commit tracked changes before submitting')
    target = STAGING_ROOT / commit
    if target.is_dir():
        if git('-C', str(target), 'rev-parse', 'HEAD') != commit:
            raise RuntimeError('Staging target does not match requested commit')
        return target
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=commit + '.', dir=STAGING_ROOT))
    try:
        shutil.copytree(repo / '.git', tmp / '.git')
        # The checkout needs no remote; do not copy stored remote credentials
        # into the worker-accessible snapshot.
        subprocess.run(['git', '-C', str(tmp), 'config', '--remove-section', 'remote.origin'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(['git', '-C', str(tmp), 'reset', '--hard', commit], check=True,
                       stdout=subprocess.DEVNULL)
        try:
            tmp.rename(target)
        except FileExistsError:
            shutil.rmtree(tmp)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target


def normalize(desc):
    return ' '.join(re.sub(r'[#@]\S+', '', desc).split())


def payload(phase, name, staged, author, nick):
    command = [str(PYTHON), '-m', 'scripts.pilot_run', phase]
    if phase != 'prepare':
        command += ['--filter', name]
    return dict(
        script=f'set -eu\ncd {shlex.quote(str(staged))}\n' + shlex.join(command),
        job_desc=f'sps-pilot {phase} {name} #{author} @{nick}',
        env_variables={'PYTHONPATH': f'{staged}:{staged}/src',
                       'HF_HOME': str(HF_HOME), 'PYTHONUNBUFFERED': '1',
                       'CC': '/usr/bin/gcc',
                       'C_INCLUDE_PATH': f'{HEADER_ROOT}/usr/include/python3.10:{HEADER_ROOT}/usr/include',
                       'OMP_NUM_THREADS': '8', 'TOKENIZERS_PARALLELISM': 'false'},
        instance_type=INSTANCE_TYPES_BY_NUM_GPUS[4], region='SR004',
        type='binary_exp', shm_size_class='large', base_image=BASE_IMAGE,
        n_workers=1, processes_per_worker=1)


def main(default_phase='train'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('prepare', 'train', 'eval'), default=default_phase)
    parser.add_argument('--author_name', default='d.tarasov')
    parser.add_argument('--telegram_nick', default='mrsndmn')
    parser.add_argument('--profile', default='default')
    parser.add_argument('--filter')
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()
    repo = Path(git('rev-parse', '--show-toplevel'))
    commit = git('rev-parse', 'HEAD')
    candidates = [('data-and-probes', ARTIFACTS_ROOT / 'prepared.json')] if args.phase == 'prepare' else [
        (exp.name, output_dir(exp) / ('trained.json' if args.phase == 'train' else 'evaluated.json'))
        for exp in collect_experiments(args.filter)]
    if not candidates:
        raise ValueError('No matching experiments')
    if not args.dry:
        if args.phase == 'train' and not (ARTIFACTS_ROOT / 'prepared.json').exists():
            raise RuntimeError('Complete --phase prepare and inspect measured estimates first')
        # mls is an optional submission-host dependency; dry plans need neither
        # it, credentials nor an API connection.
        from mls.manager.job.utils import training_job_api_from_profile, get_in_progress_jobs
        client, extra = training_job_api_from_profile(args.profile)
        active = get_in_progress_jobs(client_profile=args.profile)
        active_by_desc = {normalize(j['job_desc']): j['job_name'] for j in active}
        staged = stage_repo(repo, commit)
    else:
        staged = STAGING_ROOT / commit
        extra = {'region': 'SR004'}
        active_by_desc = {}
    jobs = []
    for name, marker in candidates:
        if marker.exists():
            if json.loads(marker.read_text()).get('design_signature') != design_signature():
                raise RuntimeError(f'Stale completion marker at {marker}; use a new artifact root/run name')
            print(f'Skip completed: {name}')
            continue
        if args.phase == 'eval' and not (output_dir(next(
                e for e in collect_experiments() if e.name == name)) / 'trained.json').exists():
            print(f'Skip evaluation without trained checkpoint: {name}')
            continue
        body = payload(args.phase, name, staged, args.author_name, args.telegram_nick)
        body['region'] = extra['region']
        identity = normalize(body['job_desc'])
        if identity in active_by_desc:
            jobs.append(dict(name=name, job_name=active_by_desc[identity], reused=True))
            continue
        if args.dry:
            print(json.dumps(body, indent=2))
            continue
        # Single submission attempt: never hold an ephemeral shell retrying a
        # quota failure. The caller may retry after the allocation changes.
        result = client.run_job(body)
        job_name = result.get('job_name')
        if not job_name or not re.fullmatch(r'lm-mpi-job-[0-9a-f-]{36}', job_name):
            raise RuntimeError(f'MLSpace rejected submission: {result}')
        uuid.UUID(job_name.removeprefix('lm-mpi-job-'))
        jobs.append(dict(name=name, job_name=job_name, commit=commit))
        active_by_desc[identity] = job_name
        receipt = ARTIFACTS_ROOT / 'submissions' / f'{job_name}.json'
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(dict(phase=args.phase, **jobs[-1]), indent=2))
        print(json.dumps(jobs[-1]))
    print('__MLS_JOBS_JSON__')
    print(json.dumps(dict(phase=args.phase, jobs=jobs)))
    return 0
