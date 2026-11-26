"""Isolated MLS jobs for step-384 eval and scheduled instruct training/eval."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
from scripts import extend_common as sc, extend_config as c
from scripts import mlspace_launch as launch


def experiments(pattern=None):
    return [x for x in sc.e.collect_experiments(pattern) if x.trainable]


def payload(variant, phase, exp, workload, staged, author, nick):
    body = launch.payload(phase,exp.name,staged,author,nick)
    command = [str(sc.e.PYTHON),'-m','scripts.extend_run','--worker','--variant',variant,
               '--phase',phase,'--filter',exp.name]
    if workload:command += ['--workload',workload]
    body['script'] = f'cd {shlex.quote(str(staged))} && ' + shlex.join(command)
    body['job_desc'] = f'sps-{c.ROOT_NAMES[variant]} {phase} {sc.arm_name(exp,workload)} step{variant} #{author} @{nick}'
    return body


def hours(phase,workload):
    return c.TRAIN_HOURS if phase=='train' else c.WORKLOAD_HOURS[workload]


def reserve(variant,phase,name,workload):
    path=sc.root(variant)/'budget.json'
    ledger=json.loads(path.read_text()) if path.exists() else {'attempts':[]}
    cost=hours(phase,workload)*(4 if phase=='train' else 1)
    if sum(x['gpu_hours'] for x in ledger['attempts'])+cost>c.BUDGETS[variant]:
        raise RuntimeError('Rerun GPU-hour reservation cap exceeded')
    ledger['attempts'].append(dict(phase=phase,name=name,gpu_hours=cost,time=time.time()))
    launch.store_json(path,ledger)


def submit(args):
    exps=experiments(args.filter)
    if not exps:raise ValueError('No matching trainable experiments')
    commit=launch.git('rev-parse','HEAD');staged=sc.e.STAGING_ROOT/commit
    active={};client=None;extra={}
    workloads=[None] if args.phase=='train' else ([args.workload] if args.workload else c.WORKLOADS)
    if not args.dry:
        sc.source_plan()
        for exp in exps:
            sc.validate_checkpoint(args.variant,exp,source=args.phase=='train')
        from mls.manager.job.utils import training_job_api_from_profile,get_in_progress_jobs
        client,extra=training_job_api_from_profile(args.profile)
        launch.disable_submission_retries(client)
        active={launch.normalize(j['job_desc']):j['job_name'] for j in get_in_progress_jobs(client_profile=args.profile)}
        launch.reconcile_receipts(client,active);launch.reconcile_intents(active)
        staged=launch.stage_repo(Path(launch.git('rev-parse','--show-toplevel')),commit)
    jobs=[]
    for exp in exps:
        for workload in workloads:
            name=sc.arm_name(exp,workload)+(f'--step{args.variant}' if workload else '')
            marker=sc.output_dir(args.variant,exp,workload)/('trained.json' if args.phase=='train' else 'evaluated.json')
            if marker.exists():
                if json.loads(marker.read_text())['design_signature']!=sc.signature():
                    raise RuntimeError('Stale rerun completion marker')
                print('Skip completed:',name);continue
            body=payload(args.variant,args.phase,exp,workload,staged,args.author_name,args.telegram_nick)
            identity=launch.normalize(body['job_desc'])
            if identity in active:
                jobs.append(dict(name=name,job_name=active[identity],reused=True));continue
            if args.dry:print(json.dumps(body,indent=2));continue
            body['region']=extra['region'];reserve(args.variant,args.phase,name,workload)
            job=launch.submit_payload(client,body,args.phase,name,commit)
            jobs.append(job);active[identity]=job['job_name'];print(json.dumps(job),flush=True)
    print('__MLS_JOBS_JSON__');print(json.dumps(dict(phase=args.phase,variant=args.variant,jobs=jobs)),flush=True)


def worker(args):
    exps=[x for x in experiments() if x.name==args.filter]
    if len(exps)!=1:raise ValueError('Select exactly one experiment')
    exp=exps[0]
    if args.phase=='eval' and not args.workload:raise ValueError('Select one eval workload')
    if args.internal:
        if args.phase!='eval':raise ValueError('Training uses torchrun')
        from scripts.extend_eval import evaluate
        return evaluate(args.variant,exp,args.workload,args.limit)
    sc.source_plan()
    if args.phase=='train':
        command=[str(sc.e.PYTHON),'-m','torch.distributed.run','--standalone',
                 f'--nproc_per_node={exp.num_gpus}','-m','scripts.extend_train','--filter',exp.name]
    else:
        command=[str(sc.e.PYTHON),'-m','scripts.extend_run','--worker','--internal',
                 '--variant',args.variant,'--phase','eval','--filter',exp.name,'--workload',args.workload]
        if args.limit:command+=['--limit',str(args.limit)]
    process=subprocess.Popen(command,start_new_session=True)
    try:
        code=process.wait(timeout=hours(args.phase,args.workload)*3600)
        if code:raise RuntimeError(f'Rerun worker exited {code}')
    except BaseException:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait();raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=tuple(c.ROOT_NAMES),default=str(c.TRAIN_STEPS))
    parser.add_argument('--phase',choices=('train','eval'),required=True)
    parser.add_argument('--workload',choices=c.WORKLOADS)
    parser.add_argument('--filter')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--internal',action='store_true')
    parser.add_argument('--limit',type=int);parser.add_argument('--dry',action='store_true')
    parser.add_argument('--profile',default='default');parser.add_argument('--author_name',default='d.tarasov')
    parser.add_argument('--telegram_nick',default='mrsndmn')
    args=parser.parse_args()
    if args.phase=='train' and (args.variant!=str(c.TRAIN_STEPS) or args.workload or args.limit):
        parser.error('Training selects final target step without eval workload or limit')
    if args.limit is not None and args.limit<1:parser.error('Positive limit required')
    if args.worker:return worker(args)
    if args.internal or args.limit:parser.error('--internal/--limit require --worker')
    launch.ARTIFACTS_ROOT=sc.root(args.variant);launch.RUN_ROOT=launch.ARTIFACTS_ROOT/'runs'
    if args.dry:return submit(args)
    with launch.submission_lock():return submit(args)


if __name__=='__main__':main()
