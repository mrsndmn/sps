"""Separate, resumable one-GPU IFEval jobs using the existing instruct protocol."""
from experiments.family_cli import worker_child
from orchestration.named.publication import work_path
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from experiments import ifeval_config as c
os.environ.setdefault('SPS_EXPERIMENT_SUITE', c.family_config.SOURCE_SUITE)
from orchestration import experiment_suite as e
from orchestration import mlspace_launch as launch

ROOT = e.COMMON_ROOT / c.family_config.ARTIFACT_NAME
RUN_ROOT = ROOT / 'runs'


def source_check(exp, *, checkpoint=False):
    if e.SUITE != c.family_config.SOURCE_SUITE:
        raise RuntimeError('IFEval requires the pinned instruct design')
    plan = json.loads((e.prepared_path()).read_text())
    if exp.trainable:
        marker = json.loads((e.output_dir(exp)/'trained.json').read_text())
        if checkpoint:
            from workloads.instruction.instruct_common import validate_final
            validate_final(e.output_dir(exp)/'final.pt', plan)
    return plan


def signature():
    return c.family_config.ARTIFACT_NAME


def arm_name(exp, mode=None):
    return exp.name if mode is None else f'{exp.name}--{mode}'


def output_dir(exp, mode=None):
    return work_path(RUN_ROOT / 'out' / arm_name(exp, mode))


def payload(exp, staged, author, nick, mode=None):
    body = launch.payload('eval', exp.name, staged, author, nick)
    command = [str(e.PYTHON), 'scripts/instruction/ifeval_run.py', '--worker', '--filter', exp.name]
    if mode:
        command += ['--mode', mode]
    body['script'] = f'cd {shlex.quote(str(staged))} && ' + shlex.join(command)
    body['job_desc'] = f'sps-{c.family_config.ARTIFACT_NAME} eval {arm_name(exp, mode)} #{author} @{nick}'
    return body


def reserve(name):
    path = ROOT/'budget.json'
    ledger = json.loads(path.read_text()) if path.exists() else {'attempts': []}
    if sum(x['gpu_hours'] for x in ledger['attempts']) + c.family_config.EVAL_HOURS > c.family_config.GPU_HOURS:
        raise RuntimeError('Parallel IFEval GPU-hour reservation cap exceeded')
    ledger['attempts'].append(dict(name=name, gpu_hours=c.family_config.EVAL_HOURS, time=time.time()))
    launch.store_json(path, ledger)


def submit(args):
    experiments = e.collect_experiments(args.filter)
    if not experiments:
        raise ValueError('No matching experiments')
    commit = launch.git('rev-parse', 'HEAD')
    staged = e.STAGING_ROOT / commit
    active = {}; client = None; extra = {}
    if not args.dry:
        modes = [args.mode] if args.mode else c.family_config.MODES
        pending = {(exp.name, mode) for exp in experiments for mode in modes
                   if launch.pending(output_dir(exp, mode), args.profile)}
        experiments = [exp for exp in experiments if any((exp.name, mode) in pending for mode in modes)]
        if not experiments:
            return
        for exp in experiments:
            source_check(exp)
        from mls.manager.job.utils import training_job_api_from_profile, get_in_progress_jobs
        client, extra = training_job_api_from_profile(args.profile)
        launch.disable_submission_retries(client)
        active = {launch.normalize(x['job_desc']): x['job_name']
                  for x in get_in_progress_jobs(client_profile=args.profile)}
        launch.reconcile_for_outputs(client, active, [output_dir(exp, mode) for exp in experiments for mode in modes if (exp.name, mode) in pending])
        staged = launch.stage_repo(Path(launch.git('rev-parse', '--show-toplevel')), commit)
    jobs = []
    for exp, mode in ((x, m) for x in experiments for m in ([args.mode] if args.mode else c.family_config.MODES)):
        if not args.dry and (exp.name, mode) not in pending:
            continue
        marker = output_dir(exp, mode)/'evaluated.json'
        if not args.dry and marker.exists():
            print('Skip completed:', exp.name)
            continue
        body = payload(exp, staged, args.author_name, args.telegram_nick, mode)
        identity = launch.output_identity(body['job_desc'], marker.parent)
        if identity in active:
            jobs.append(dict(name=arm_name(exp, mode), job_name=active[identity], reused=True))
            continue
        if args.dry:
            launch.print_preview(body, marker.parent, staged, "eval")
            continue
        body['region'] = extra['region']
        reserve(arm_name(exp, mode))
        job = launch.submit_payload(client, body, 'eval', arm_name(exp, mode), commit, output_dir=marker.parent)
        jobs.append(job)
        if job.get('job_name'):
            active[identity] = job['job_name']
        print(json.dumps(job), flush=True)
    if args.dry:
        return
    print('__MLS_JOBS_JSON__')
    print(json.dumps(dict(phase='ifeval', jobs=jobs)), flush=True)


def evaluate(exp, limit=None, mode=None):
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from langdetect import DetectorFactory
    from experiments import instruct_config as source
    from workloads.instruction.instruct_common import load_model, restore
    from workloads.instruction.instruct_evaluate import ensure_ifeval_resources, evaluate_generation
    source_check(exp, checkpoint=True)
    DetectorFactory.seed = source.family_config.SEED
    ensure_ifeval_resources()
    gate = json.loads((e.SANITY_ROOT/'generation-protocol.json').read_text())
    if gate['generation'] != source.family_config.GENERATION:
        raise RuntimeError('Stale source generation audit')
    original = json.loads((e.output_dir(exp)/'eval/manifest.json').read_text())
    if (original['lm_eval_version'] != importlib.metadata.version('lm_eval')):
        raise RuntimeError('IFEval scoring implementation differs from original evaluation')
    root = output_dir(exp, mode) / ('smoke' if limit else 'eval')
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(design_signature=signature(), source_suite=c.family_config.SOURCE_SUITE,
                    experiment=exp.__dict__, generation=source.family_config.GENERATION, modes=(mode,) if mode else c.family_config.MODES,
                    dataset=source.family_config.IFEVAL_DATASET, revision=source.family_config.IFEVAL_REVISION,
                    lm_eval_version=original["lm_eval_version"], limit=limit)
    # JSON round-trip normalizes tuples for exact resume comparison.
    manifest = json.loads(json.dumps(manifest))
    if exp.trainable:
        path = e.output_dir(exp)/'final.pt'; stat = path.stat()
        manifest['checkpoint'] = dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    meta = root/'manifest.json'
    if meta.exists() and json.loads(meta.read_text()) != manifest:
        raise RuntimeError('Parallel IFEval resume provenance mismatch')
    launch.store_json(meta, manifest)
    model = load_model(exp.recipe, 'cuda').eval().requires_grad_(False)
    if exp.trainable:
        restore(e.output_dir(exp)/'final.pt', model, 'cuda')
    tokenizer = AutoTokenizer.from_pretrained(source.family_config.MODEL, revision=source.family_config.REVISION)
    rows = list(load_dataset(source.family_config.IFEVAL_DATASET, revision=source.family_config.IFEVAL_REVISION, split='train'))
    if limit:
        rows = rows[:limit]
    results = dict(design_signature=signature(), source_suite=c.family_config.SOURCE_SUITE,
                   experiment=exp.name, tasks={})
    selected_mode = mode
    started = time.monotonic()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for mode in ((mode,) if mode else c.family_config.MODES):
            results['tasks'][mode] = evaluate_generation(model, tokenizer, rows, 'ifeval', mode,
                source.family_config.GENERATION[mode], root, signature(), source.family_config.SEED, started,
                c.family_config.EVAL_HOURS - c.family_config.STARTUP_SECONDS/3600)
    launch.store_json(root/'results.json', results)
    if not limit:
        launch.store_json(output_dir(exp, selected_mode)/'evaluated.json', results)
    print(json.dumps(results), flush=True)


def worker(args):
    matches = [x for x in e.collect_experiments() if x.name == args.filter]
    if len(matches) != 1:
        raise ValueError('Select exactly one IFEval arm')
    if args.internal:
        return evaluate(matches[0], args.limit, args.mode)
    command = [str(e.PYTHON), 'scripts/instruction/ifeval_run.py', '--worker', '--internal', '--filter', args.filter]
    if args.mode:
        command += ['--mode', args.mode]
    if args.limit:
        command += ['--limit', str(args.limit)]
    process = subprocess.Popen(worker_child(command), start_new_session=True)
    try:
        code = process.wait(timeout=c.family_config.EVAL_HOURS*3600)
        if code:
            raise RuntimeError(f'IFEval worker exited {code}')
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter')
    parser.add_argument('--mode', choices=c.family_config.MODES)
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--internal', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--profile', default='default')
    parser.add_argument('--author_name', default='d.tarasov')
    parser.add_argument('--telegram_nick', default='mrsndmn')
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    if e.SUITE != c.family_config.SOURCE_SUITE:
        parser.error('Select the instruct source suite')
    if args.worker:
        if not args.mode:
            parser.error('Workers require --mode to prevent duplicate sequential evaluation')
        return worker(args)
    if args.internal or args.limit:
        parser.error('--internal/--limit require --worker')
    # Reuse the proven locks/receipts, with a separate namespace and success paths.
    launch.ARTIFACTS_ROOT = ROOT
    launch.RUN_ROOT = RUN_ROOT
    if args.dry:
        return submit(args)
    with launch.submission_lock():
        return submit(args)
