"""Submit one durable prepare+train job; data and resume state share an isolated root."""
from experiments.family_cli import worker_child
from experiments.launch import run_filter
import argparse
import json
from pathlib import Path
import os
import shlex
import signal
import subprocess
from workloads.adaptation import base50_config as config

e, c = config.configure()
from orchestration import mlspace_launch as launch


def payload(exp, staged):
    body = launch.payload('train', exp.name, staged, 'd.tarasov', 'mrsndmn', training_gpus=exp.num_gpus)
    command = [str(e.PYTHON), 'scripts/adaptation/base50_run.py', '--worker', '--filter', exp.name]
    body['script'] = f'cd {shlex.quote(str(staged))} && '+shlex.join(command)
    body['job_desc'] = f'sps-{config.ROOT_NAME} train {exp.name} #d.tarasov @mrsndmn'
    return body


def plan():
    return dict(steps=config.STEPS, stage_steps=config.STAGE_STEPS,
                text_tokens=config.STEPS*c.family_config.GLOBAL_BATCH*c.family_config.CONTEXT,
                design_signature=e.SUITE)


def worker(exp):
    from data.ar_sps_data import prepare
    prepare()
    launch.store_json(e.prepared_path(), plan())
    command = [str(e.PYTHON), '-m', 'torch.distributed.run', '--standalone',
               f'--nproc_per_node={exp.num_gpus}', 'scripts/adaptation/base50_train.py', '--filter', exp.name]
    subprocess.run(worker_child(command), check=True)


def submit(exp, dry):
    commit = launch.git('rev-parse', 'HEAD')
    staged = e.STAGING_ROOT/commit
    marker = e.RUN_ROOT/'out'/exp.name/'trained.json'
    if not dry and not launch.pending(marker.parent):
        return
    if not dry and marker.exists():
        print('Already completed'); return
    if dry:
        launch.print_preview(payload(exp, staged), marker.parent, staged, "train"); return
    from mls.manager.job.utils import training_job_api_from_profile, get_in_progress_jobs
    client, extra = training_job_api_from_profile('default')
    launch.disable_submission_retries(client)
    active = {launch.normalize(j['job_desc']):j['job_name'] for j in get_in_progress_jobs(client_profile='default')}
    launch.reconcile_for_outputs(client, active, [marker.parent])
    identity = launch.output_identity(payload(exp, staged)['job_desc'], marker.parent)
    if identity in active:
        print(json.dumps(dict(job_name=active[identity], reused=True))); return
    staged = launch.stage_repo(Path(launch.git('rev-parse', '--show-toplevel')), commit)
    body = payload(exp, staged); body['region'] = extra['region']
    launch.store_json(e.ARTIFACTS_ROOT/'followup.json', dict(reminder=config.REMINDER,
                     checkpoint=str(e.RUN_ROOT/'out'/exp.name/'final.pt')))
    result = launch.submit_payload(client, body, 'train', exp.name, commit, output_dir=marker.parent)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--internal', action='store_true')
    configs = run_filter(parser, config.experiment_configs)
    args = parser.parse_args()
    exp = configs[args.filter]
    if args.internal:
        if not args.worker: parser.error('--internal requires --worker')
        return worker(exp)
    if args.worker:
        process = subprocess.Popen(worker_child([str(e.PYTHON), 'scripts/adaptation/base50_run.py', '--worker', '--internal', '--filter', exp.name]), start_new_session=True)
        try:
            code = process.wait(timeout=config.JOB_HOURS*3600)
            if code: raise RuntimeError(f'Long base worker exited {code}')
        except BaseException:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait(); raise
        return
    if args.dry: return submit(exp, True)
    with launch.submission_lock(): submit(exp, False)
