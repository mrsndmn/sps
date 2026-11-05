"""Separate, resumable one-GPU IFEval jobs using the existing instruct protocol."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from scripts import ifeval_config as c
os.environ.setdefault('SPS_EXPERIMENT_SUITE', c.SOURCE_SUITE)
import experiments as e
from scripts import mlspace_launch as launch

ROOT = e.COMMON_ROOT / c.ARTIFACT_NAME
RUN_ROOT = ROOT / 'runs'


def source_check(exp, *, checkpoint=False):
    if e.SUITE != c.SOURCE_SUITE or e.design_signature() != c.SOURCE_SIGNATURE:
        raise RuntimeError('IFEval requires the pinned instruct design')
    plan = json.loads((e.ARTIFACTS_ROOT/'prepared.json').read_text())
    if plan['design_signature'] != c.SOURCE_SIGNATURE:
        raise RuntimeError('Stale source preparation')
    if exp.trainable:
        marker = json.loads((e.output_dir(exp)/'trained.json').read_text())
        if marker['design_signature'] != c.SOURCE_SIGNATURE:
            raise RuntimeError('Stale source training marker')
        if checkpoint:
            from scripts.instruct_common import validate_final
            validate_final(e.output_dir(exp)/'final.pt', plan)
    return plan


def signature():
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
               for name in ('ifeval_run.py', 'ifeval_config.py')}
    return hashlib.sha256(json.dumps(dict(source=e.design_signature(), files=sources), sort_keys=True).encode()).hexdigest()


def output_dir(exp):
    return RUN_ROOT / 'out' / exp.name


def payload(exp, staged, author, nick):
    body = launch.payload('eval', exp.name, staged, author, nick)
    command = [str(e.PYTHON), '-m', 'scripts.ifeval_run', '--worker', '--filter', exp.name]
    body['script'] = f'cd {shlex.quote(str(staged))} && ' + shlex.join(command)
    body['job_desc'] = f'sps-{c.ARTIFACT_NAME} eval {exp.name} #{author} @{nick}'
    return body


def reserve(name):
    path = ROOT/'budget.json'
    ledger = json.loads(path.read_text()) if path.exists() else {'attempts': []}
    if sum(x['gpu_hours'] for x in ledger['attempts']) + c.EVAL_HOURS > c.GPU_HOURS:
        raise RuntimeError('Parallel IFEval GPU-hour reservation cap exceeded')
    ledger['attempts'].append(dict(name=name, gpu_hours=c.EVAL_HOURS, time=time.time()))
    launch.store_json(path, ledger)


def submit(args):
    experiments = e.collect_experiments(args.filter)
    if not experiments:
        raise ValueError('No matching experiments')
    commit = launch.git('rev-parse', 'HEAD')
    staged = e.STAGING_ROOT / commit
    active = {}; client = None; extra = {}
    if not args.dry:
        for exp in experiments:
            source_check(exp)
        from mls.manager.job.utils import training_job_api_from_profile, get_in_progress_jobs
        client, extra = training_job_api_from_profile(args.profile)
        launch.disable_submission_retries(client)
        active = {launch.normalize(x['job_desc']): x['job_name']
                  for x in get_in_progress_jobs(client_profile=args.profile)}
        launch.reconcile_receipts(client, active)
        launch.reconcile_intents(active)
        staged = launch.stage_repo(Path(launch.git('rev-parse', '--show-toplevel')), commit)
    jobs = []
    for exp in experiments:
        marker = output_dir(exp)/'evaluated.json'
        if marker.exists():
            if json.loads(marker.read_text())['design_signature'] != signature():
                raise RuntimeError('Stale IFEval completion marker')
            print('Skip completed:', exp.name)
            continue
        body = payload(exp, staged, args.author_name, args.telegram_nick)
        identity = launch.normalize(body['job_desc'])
        if identity in active:
            jobs.append(dict(name=exp.name, job_name=active[identity], reused=True))
            continue
        if args.dry:
            print(json.dumps(body, indent=2))
            continue
        body['region'] = extra['region']
        reserve(exp.name)
        job = launch.submit_payload(client, body, 'eval', exp.name, commit)
        jobs.append(job); active[identity] = job['job_name']
        print(json.dumps(job), flush=True)
    print('__MLS_JOBS_JSON__')
    print(json.dumps(dict(phase='ifeval', jobs=jobs)), flush=True)


def evaluate(exp, limit=None):
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from langdetect import DetectorFactory
    from lm_eval.tasks.ifeval import utils, instructions, instructions_registry, instructions_util
    from scripts import instruct_config as source
    from scripts.instruct_common import load_model, restore
    from scripts.instruct_evaluate import ensure_ifeval_resources, evaluate_generation
    source_check(exp, checkpoint=True)
    DetectorFactory.seed = source.SEED
    ensure_ifeval_resources()
    gate = json.loads((e.SANITY_ROOT/'generation-protocol.json').read_text())
    if gate['design_signature'] != c.SOURCE_SIGNATURE or gate['generation'] != source.GENERATION:
        raise RuntimeError('Stale source generation audit')
    evaluator_code = {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                      for m in (utils, instructions, instructions_registry, instructions_util)}
    original = json.loads((e.output_dir(exp)/'eval/manifest.json').read_text())
    if (original['design_signature'] != c.SOURCE_SIGNATURE
            or original['ifeval_code'] != evaluator_code
            or original['lm_eval_version'] != importlib.metadata.version('lm_eval')):
        raise RuntimeError('IFEval scoring implementation differs from original evaluation')
    root = output_dir(exp) / ('smoke' if limit else 'eval')
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(design_signature=signature(), source_signature=c.SOURCE_SIGNATURE,
                    experiment=exp.__dict__, generation=source.GENERATION, modes=c.MODES,
                    dataset=source.IFEVAL_DATASET, revision=source.IFEVAL_REVISION,
                    evaluator_code=evaluator_code, limit=limit)
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
    tokenizer = AutoTokenizer.from_pretrained(source.MODEL, revision=source.REVISION)
    rows = list(load_dataset(source.IFEVAL_DATASET, revision=source.IFEVAL_REVISION, split='train'))
    if limit:
        rows = rows[:limit]
    results = dict(design_signature=signature(), source_signature=c.SOURCE_SIGNATURE,
                   experiment=exp.name, tasks={})
    started = time.monotonic()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for mode in c.MODES:
            results['tasks'][mode] = evaluate_generation(model, tokenizer, rows, 'ifeval', mode,
                source.GENERATION[mode], root, signature(), source.SEED, started,
                c.EVAL_HOURS - c.STARTUP_SECONDS/3600)
    launch.store_json(root/'results.json', results)
    if not limit:
        launch.store_json(output_dir(exp)/'evaluated.json', results)
    print(json.dumps(results), flush=True)


def worker(args):
    matches = [x for x in e.collect_experiments() if x.name == args.filter]
    if len(matches) != 1:
        raise ValueError('Select exactly one IFEval arm')
    if args.internal:
        return evaluate(matches[0], args.limit)
    command = [str(e.PYTHON), '-m', 'scripts.ifeval_run', '--worker', '--internal', '--filter', args.filter]
    if args.limit:
        command += ['--limit', str(args.limit)]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        code = process.wait(timeout=c.EVAL_HOURS*3600)
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
    if e.SUITE != c.SOURCE_SUITE:
        parser.error('Select the instruct source suite')
    if args.worker:
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


if __name__ == '__main__':
    main()
