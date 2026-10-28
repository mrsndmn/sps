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
import hashlib
from contextlib import contextmanager
import socket
import time

from experiments import (ARTIFACTS_ROOT, RUN_ROOT, STAGING_ROOT, HF_HOME,
                        PYTHON, BASE_IMAGE, INSTANCE_TYPES_BY_NUM_GPUS,
                        collect_experiments, output_dir, design_signature, SUITE, training_marker, EVAL_NUM_GPUS)
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


@contextmanager
def submission_lock():
    """NFS mkdir is atomic; fail closed rather than trusting NFS flock leases."""
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    lock = ARTIFACTS_ROOT / '.submission-lock'
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(f'Another submission holds {lock}; inspect owner.json before removing a stale lock')
    try:
        (lock / 'owner.json').write_text(json.dumps(dict(host=socket.gethostname(),
                                                       pid=os.getpid(), started=time.time())))
        yield
    finally:
        shutil.rmtree(lock)


def reconcile_receipts(client, active_by_desc):
    # A just-submitted job may not yet appear in the list API. Its durable
    # receipt prevents a second submission; only a confirmed failed/cancelled
    # state permits retry. Unknown or unreachable states fail closed.
    for path in sorted((ARTIFACTS_ROOT / 'submissions').glob('*.json')):
        receipt = json.loads(path.read_text())
        identity = receipt['identity']
        if identity in active_by_desc:
            continue
        result = client.get_job_status(receipt['job_name'])
        status = result.get('status')
        if status is None:
            raise RuntimeError(f'Cannot establish job status for {receipt["job_name"]}')
        # The API returns lowercase states; some mls versions/title-cased
        # fixtures use uppercase initials. Reconcile both identically.
        status = str(status).strip().casefold()
        retryable = {'failed', 'stopped', 'terminated', 'deleted', 'cancelled', 'canceled'}
        if status in {'completed', 'succeeded'}:
            marker = ARTIFACTS_ROOT / 'prepared.json' if receipt['phase'] == 'prepare' else (
                RUN_ROOT / 'out' / receipt['name'] /
                ('trained.json' if receipt['phase'] == 'train' else 'evaluated.json'))
            if not marker.exists():
                raise RuntimeError(f'Job {receipt["job_name"]} completed without success marker {marker}')
            continue
        if status not in retryable:
            active_by_desc[identity] = receipt['job_name']


def store_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def disable_submission_retries(client):
    """The shared client retries POST by default; intent must cover ONE send."""
    from urllib3.util.retry import Retry
    session = client._session
    for adapter in session.adapters.values():
        adapter.max_retries = Retry(total=0, connect=0, read=0, status=0,
                                    redirect=0, other=0)
    session.max_redirects = 0


def reconcile_intents(active_by_desc):
    for path in sorted((ARTIFACTS_ROOT / 'submission-intents').glob('*.json')):
        intent = json.loads(path.read_text())
        identity = intent['identity']
        if identity not in active_by_desc:
            raise RuntimeError(f'Unresolved MLSpace submission at {path}; reconcile with job history '
                               'before clearing it. Automatic resubmission is blocked.')
        job_name = active_by_desc[identity]
        store_json(ARTIFACTS_ROOT / 'submissions' / f'{job_name}.json',
                   dict(intent, job_name=job_name))
        path.unlink()


def submit_payload(client, body, phase, name, commit):
    identity = normalize(body['job_desc'])
    intent = ARTIFACTS_ROOT / 'submission-intents' / (
        hashlib.sha256(identity.encode()).hexdigest() + '.json')
    if intent.exists():
        raise RuntimeError(f'Unresolved MLSpace submission at {intent}')
    # Persist BEFORE the API call. A lost response must never cause automatic
    # resubmission, even if the server's job list has not caught up yet.
    data = dict(phase=phase, identity=identity, name=name, commit=commit,
                attempted_at=time.time())
    store_json(intent, data)
    result = client.run_job(body)
    job_name = result.get('job_name')
    if not job_name or not re.fullmatch(r'lm-mpi-job-[0-9a-f-]{36}', job_name):
        raise RuntimeError(f'MLSpace rejected submission; reconcile intent at {intent}: {result}')
    uuid.UUID(job_name.removeprefix('lm-mpi-job-'))
    store_json(ARTIFACTS_ROOT / 'submissions' / f'{job_name}.json', dict(data, job_name=job_name))
    intent.unlink()
    return dict(name=name, job_name=job_name, commit=commit)


def payload(phase, name, staged, author, nick):
    command = [str(PYTHON), '-m', 'scripts.mtp_run' if SUITE in {'mtp', 'mtp-unfreeze'} else 'scripts.pilot_run', phase]
    if SUITE == 'spec-baselines':
        command = [str(PYTHON), '-m', 'scripts.spec_baseline_run', phase]
    if SUITE == 'mtp-unified':
        command = [str(PYTHON), '-m', 'scripts.mtp_unified_evaluate', phase]
    if SUITE == 'mtp-parallel':
        command = [str(PYTHON), '-m', 'scripts.mtp_parallel_evaluate', phase]
    if SUITE == 'ar-sps':
        command = [str(PYTHON), '-m', 'scripts.ar_sps_run', phase]
    if phase != 'prepare':
        command += ['--filter', name]
    return dict(
        # MLSpace validates script as a single shell line (newlines rejected).
        script=f'cd {shlex.quote(str(staged))} && ' + shlex.join(command),
        job_desc=f'sps-{"pilot" if SUITE == "1b" else SUITE} {phase} {name} #{author} @{nick}',
        env_variables={'PYTHONPATH': f'{staged}:{staged}/src',
                       'SPS_EXPERIMENT_SUITE': SUITE,
                       'HF_HOME': str(HF_HOME), 'PYTHONUNBUFFERED': '1',
                       'CC': '/usr/bin/gcc',
                       'CXX': '/usr/bin/g++',
                       'CPLUS_INCLUDE_PATH': f'{HEADER_ROOT}/usr/include/python3.10:{HEADER_ROOT}/usr/include',
                       'C_INCLUDE_PATH': f'{HEADER_ROOT}/usr/include/python3.10:{HEADER_ROOT}/usr/include',
                       'OMP_NUM_THREADS': '8', 'TOKENIZERS_PARALLELISM': 'false'},
        instance_type=INSTANCE_TYPES_BY_NUM_GPUS[EVAL_NUM_GPUS if phase == 'eval' else 4], region='SR004',
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
    if args.dry:
        return submit(args)
    with submission_lock():
        return submit(args)


def submit(args):
    if SUITE in {'mtp-parallel','mtp-unified'} and args.phase != 'eval':
        raise ValueError('mtp-parallel is evaluation-only')
    repo = Path(git('rev-parse', '--show-toplevel'))
    commit = git('rev-parse', 'HEAD')
    candidates = [('data-and-probes', ARTIFACTS_ROOT / 'prepared.json')] if args.phase == 'prepare' else [
        (exp.name, output_dir(exp) / ('trained.json' if args.phase == 'train' else 'evaluated.json'))
        for exp in collect_experiments(args.filter) if args.phase != 'train' or exp.trainable]
    if not candidates:
        raise ValueError('No matching experiments')
    if not args.dry:
        if args.phase == 'train' and not (ARTIFACTS_ROOT / 'prepared.json').exists():
            raise RuntimeError('Complete --phase prepare and inspect measured estimates first')
        # mls is an optional submission-host dependency; dry plans need neither
        # it, credentials nor an API connection.
        from mls.manager.job.utils import training_job_api_from_profile, get_in_progress_jobs
        client, extra = training_job_api_from_profile(args.profile)
        disable_submission_retries(client)
        active = get_in_progress_jobs(client_profile=args.profile)
        active_by_desc = {normalize(j['job_desc']): j['job_name'] for j in active}
        reconcile_receipts(client, active_by_desc)
        reconcile_intents(active_by_desc)
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
        train_marker = training_marker(next(e for e in collect_experiments() if e.name == name)) if args.phase == 'eval' else None
        if args.phase == 'eval' and train_marker is not None and not train_marker.exists():
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
        if SUITE == 'ar-sps':
            from scripts.ar_sps_run import reserve
            reserve(args.phase, name)
        jobs.append(submit_payload(client, body, args.phase, name, commit))
        active_by_desc[identity] = jobs[-1]['job_name']
        print(json.dumps(jobs[-1]))
    print('__MLS_JOBS_JSON__')
    print(json.dumps(dict(phase=args.phase, jobs=jobs)))
    return 0
