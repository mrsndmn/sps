import sys
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

from orchestration.experiment_suite import (ARTIFACTS_ROOT, RUN_ROOT, STAGING_ROOT, HF_HOME,
                        PYTHON, BASE_IMAGE, INSTANCE_TYPES_BY_NUM_GPUS,
                        collect_experiments, output_dir, SUITE, training_marker, EVAL_NUM_GPUS)
from orchestration.pilot_headers import HEADER_ROOT
from orchestration.suite_registry import AR_SPS_SUITES, EVAL_ONLY_SUITES, INSTRUCT_SUITE, WORKER_SCRIPTS


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
        git_dir = Path(git('-C', str(repo), 'rev-parse', '--path-format=absolute', '--git-common-dir'))
        shutil.copytree(git_dir, tmp / '.git', ignore=shutil.ignore_patterns('worktrees'))
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


def receipt_outputs(record):
    if output := record.get("output_dir"):
        return {Path(output)}
    return set()


def reconcile_for_outputs(client, active, outputs):
    selected = set(outputs)
    reconcile_receipts(client, active, selected)
    reconcile_intents(active, selected)


def occupied_slots(client, active, prefix):
    """Count family submissions during list-API lag, including unconfirmed sends."""
    from orchestration.pilot_launch import TERMINAL_FAILURES

    listed = {key: job for key, job in active.items() if key.startswith(prefix)}
    jobs = set(listed.values())
    received_attempts = set()
    for path in (ARTIFACTS_ROOT / "submissions").glob("*.json"):
        row = json.loads(path.read_text())
        if not row["identity"].startswith(prefix):
            continue
        received_attempts.add((row["identity"], row.get("attempt")))
        job = row["job_name"]
        if job not in jobs:
            state = str(client.get_job_status(job).get("status", "")).strip().casefold()
            if state not in TERMINAL_FAILURES | {"completed", "succeeded"}:
                jobs.add(job)
    unknown = 0
    for path in (ARTIFACTS_ROOT / "submission-intents").glob("*.json"):
        row = json.loads(path.read_text())
        identity = row["identity"]
        if not identity.startswith(prefix) or identity in listed:
            continue
        if (identity, row.get("attempt")) in received_attempts:
            continue
        # The worker can finish even if the submitter never received its job ID.
        output = row.get("output_dir")
        if not output or not Path(output).is_dir():
            unknown += 1
    return len(jobs) + unknown


def reconcile_receipts(client, active_by_desc, outputs=None):
    # A just-submitted job may not yet appear in the list API. Its durable
    # receipt prevents a second submission; only a confirmed failed/cancelled
    # state permits retry. Unknown or unreachable states fail closed.
    for path in sorted((ARTIFACTS_ROOT / 'submissions').glob('*.json')):
        receipt = json.loads(path.read_text())
        if outputs is not None and not outputs.intersection(receipt_outputs(receipt)):
            continue
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
            if receipt.get("output_dir"):
                marker = Path(receipt["output_dir"])
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


def reconcile_intents(active_by_desc, outputs=None):
    for path in sorted((ARTIFACTS_ROOT / 'submission-intents').glob('*.json')):
        intent = json.loads(path.read_text())
        if outputs is not None and not outputs.intersection(receipt_outputs(intent)):
            continue
        identity = intent['identity']
        if intent.get('attempt') or identity not in active_by_desc:
            raise RuntimeError(f'Unresolved MLSpace submission at {path}; reconcile with job history '
                               'before clearing it. Automatic resubmission is blocked.')
        job_name = active_by_desc[identity]
        store_json(ARTIFACTS_ROOT / 'submissions' / f'{job_name}.json',
                   dict(intent, job_name=job_name))
        path.unlink()


def valid_job_name(result):
    name = result.get('job_name') if isinstance(result, dict) else None
    if isinstance(name, str) and re.fullmatch(r'lm-mpi-job-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', name):
        return name
    return None


def pending(output, profile="default"):
    from orchestration.named.publication import pending_output
    return pending_output(Path(os.environ.get("SPS_SUBMIT_OUTPUT") or output), profile)


def output_identity(description, output):
    key = hashlib.sha256(str(Path(output).resolve()).encode()).hexdigest()
    return f"{normalize(description)} output={key}"


def scope_payload(body, output):
    identity = output_identity(body["job_desc"], output)
    tags = re.findall(r"(?:^|\s)([#@]\S+)", body["job_desc"])
    body["job_desc"] = " ".join([identity, *tags])


def prepare_payload(body, output, staged, phase):
    """Build the exact worker invocation used for preview and submission."""
    from experiments.protocols import schemas
    from orchestration.named.publication import published_command

    destination = Path(os.environ.get("SPS_SUBMIT_OUTPUT") or output)
    body = dict(body)
    scope_payload(body, destination)
    groups = schemas(Path(sys.argv[0]).stem, phase)
    body["script"] = published_command(body["script"], destination, staged, groups, original=output)
    return body


def print_preview(body, output, staged, phase):
    payload = prepare_payload(body, output, staged, phase)
    if os.environ.get("SPS_NAMED_PREVIEW"):
        print(json.dumps(dict(payload=payload, output_dir=os.environ.get("SPS_SUBMIT_OUTPUT") or str(output)), indent=2))
    else:
        print(json.dumps(payload, indent=2))


def submit_payload(client, body, phase, name, commit, *, output_dir=None, directory=None, before_send=None):
    output = directory.output if directory else os.environ.get("SPS_SUBMIT_OUTPUT") or output_dir
    if output and directory is None:
        from orchestration.named.publication import COMMON_ROOT, RunDirectory
        body = prepare_payload(body, output_dir or output, STAGING_ROOT / commit, phase)
        directory = RunDirectory(COMMON_ROOT, Path(output))
    identity = normalize(body["job_desc"])
    intent = ARTIFACTS_ROOT / 'submission-intents' / (
        hashlib.sha256(identity.encode()).hexdigest() + '.json')
    if intent.exists():
        raise RuntimeError(f'Unresolved MLSpace submission at {intent}')
    # Persist BEFORE the API call. A lost response must never cause automatic
    # resubmission, even if the server's job list has not caught up yet.
    data = dict(phase=phase, identity=identity, name=name, commit=commit,
                attempted_at=time.time())
    if directory:
        if not directory.reserve():
            return dict(name=name, reused=True, output_dir=str(output))
        attempt = uuid.uuid4().hex
        body["job_desc"] += f" #attempt-{attempt}"
        data.update(output_dir=str(output), attempt=attempt)
    try:
        if directory:
            store_json(directory.reservation / "submission.json", dict(
                identity=identity, attempt=attempt, receipts=str(ARTIFACTS_ROOT / "submissions"), intent=str(intent)
            ))
        if before_send:
            before_send()
        store_json(intent, data)
    except Exception:
        if directory:
            directory.retire()
        raise
    result = client.run_job(body)
    job_name = valid_job_name(result)
    if directory and not job_name and isinstance(result, dict) and result.get('error_code') == 1:
        directory.retire()
        intent.unlink()
        raise RuntimeError(f'MLSpace rejected submission: {result}')
    if directory and job_name:
        try:
            directory.record(job_name)
        except FileNotFoundError:
            if not directory.output.is_dir():
                raise
    if not job_name or not re.fullmatch(r'lm-mpi-job-[0-9a-f-]{36}', job_name):
        raise RuntimeError(f'MLSpace rejected submission; reconcile intent at {intent}: {result}')
    uuid.UUID(job_name.removeprefix('lm-mpi-job-'))
    store_json(ARTIFACTS_ROOT / 'submissions' / f'{job_name}.json', dict(data, job_name=job_name))
    intent.unlink()
    return dict(name=name, job_name=job_name, commit=commit)


def payload(phase, name, staged, author, nick, *, training_gpus=4):
    command = [str(PYTHON), WORKER_SCRIPTS[SUITE], phase]
    if phase != 'prepare':
        command += ['--filter', name]
    return dict(
        # MLSpace validates script as a single shell line (newlines rejected).
        script=f'cd {shlex.quote(str(staged))} && ' + shlex.join(command),
        job_desc=f'sps-{"pilot" if SUITE == "1b" else SUITE} {phase} {name} #{author} @{nick}',
        env_variables={**({'HF_HUB_DISABLE_XET': '1'} if SUITE == INSTRUCT_SUITE else {}),
                       'PYTHONPATH': f'{staged}/src',
                       'SPS_EXPERIMENT_SUITE': SUITE,
                       'HF_HOME': str(HF_HOME), 'PYTHONUNBUFFERED': '1',
                       'CC': '/usr/bin/gcc',
                       'CXX': '/usr/bin/g++',
                       'CPLUS_INCLUDE_PATH': f'{HEADER_ROOT}/usr/include/python3.10:{HEADER_ROOT}/usr/include',
                       'C_INCLUDE_PATH': f'{HEADER_ROOT}/usr/include/python3.10:{HEADER_ROOT}/usr/include',
                       'OMP_NUM_THREADS': '8', 'TOKENIZERS_PARALLELISM': 'false'},
        instance_type=INSTANCE_TYPES_BY_NUM_GPUS[EVAL_NUM_GPUS if phase == 'eval' else training_gpus], region='SR004',
        type='binary_exp', shm_size_class='large', base_image=BASE_IMAGE,
        n_workers=1, processes_per_worker=1)


def main(default_phase='train', *, named=None):
    parser = argparse.ArgumentParser(epilog="Use --list [FAMILY] to browse named runs. "
                                     "Select one with --experiment FAMILY/NAME.")
    parser.add_argument('--phase', choices=('prepare', 'train', 'eval'), default=default_phase)
    parser.add_argument('--author_name', default='d.tarasov')
    parser.add_argument('--telegram_nick', default='mrsndmn')
    parser.add_argument('--profile', default='default')
    parser.add_argument('--filter')
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()
    args.named = named
    return submit(args)


def submit(args):
    from orchestration.pilot_launch import submit as submit_directory
    return submit_directory(args)
