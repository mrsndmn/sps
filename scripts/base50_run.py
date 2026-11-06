"""Submit one durable prepare+train job; data and resume state share an isolated root."""
import argparse
import json
from pathlib import Path
import os
import shlex
import signal
import subprocess
from scripts import base50_config as config

e, c = config.configure()
from scripts import mlspace_launch as launch


def payload(exp, staged):
    body = launch.payload('train', exp.name, staged, 'd.tarasov', 'mrsndmn')
    command = [str(e.PYTHON), '-m', 'scripts.base50_run', '--worker']
    body['script'] = f'cd {shlex.quote(str(staged))} && '+shlex.join(command)
    body['job_desc'] = f'sps-{config.ROOT_NAME} train {exp.name} #d.tarasov @mrsndmn'
    return body


def plan():
    return dict(steps=config.STEPS, stage_steps=config.STAGE_STEPS,
                text_tokens=config.STEPS*c.GLOBAL_BATCH*c.CONTEXT,
                design_signature=e.design_signature())


def worker():
    from scripts.ar_sps_data import prepare
    prepare()
    launch.store_json(e.ARTIFACTS_ROOT/'prepared.json', plan())
    command = [str(e.PYTHON), '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=4', '-m', 'scripts.base50_train', '--filter', c.experiment_configs[0].name]
    subprocess.run(command, check=True)


def submit(dry):
    commit = launch.git('rev-parse', 'HEAD')
    staged = e.STAGING_ROOT/commit
    exp = c.experiment_configs[0]
    marker = e.RUN_ROOT/'out'/exp.name/'trained.json'
    if marker.exists():
        if json.loads(marker.read_text())['design_signature'] != e.design_signature():
            raise RuntimeError('Stale terminal marker')
        print('Already completed'); return
    if dry:
        print(json.dumps(payload(exp, staged), indent=2)); return
    from mls.manager.job.utils import training_job_api_from_profile, get_in_progress_jobs
    client, extra = training_job_api_from_profile('default')
    launch.disable_submission_retries(client)
    active = {launch.normalize(j['job_desc']):j['job_name'] for j in get_in_progress_jobs(client_profile='default')}
    launch.reconcile_receipts(client, active); launch.reconcile_intents(active)
    identity = launch.normalize(payload(exp, staged)['job_desc'])
    if identity in active:
        print(json.dumps(dict(job_name=active[identity], reused=True))); return
    staged = launch.stage_repo(Path(launch.git('rev-parse', '--show-toplevel')), commit)
    body = payload(exp, staged); body['region'] = extra['region']
    launch.store_json(e.ARTIFACTS_ROOT/'followup.json', dict(reminder=config.REMINDER,
                     checkpoint=str(e.RUN_ROOT/'out'/exp.name/'final.pt')))
    result = launch.submit_payload(client, body, 'train', exp.name, commit)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--internal', action='store_true')
    args = parser.parse_args()
    if args.internal:
        if not args.worker: parser.error('--internal requires --worker')
        return worker()
    if args.worker:
        process = subprocess.Popen([str(e.PYTHON), '-m', 'scripts.base50_run', '--worker', '--internal'], start_new_session=True)
        try:
            code = process.wait(timeout=config.JOB_HOURS*3600)
            if code: raise RuntimeError(f'Long base worker exited {code}')
        except BaseException:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait(); raise
        return
    if args.dry: return submit(True)
    with launch.submission_lock(): submit(False)


if __name__ == '__main__':
    main()
