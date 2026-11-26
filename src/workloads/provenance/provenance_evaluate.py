"""Fixed final-checkpoint IFEval and crossed held-out NLL for provenance controls."""
from orchestration.named.publication import work_path
import argparse
from functools import lru_cache
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import time
from unittest.mock import patch

from workloads.provenance import provenance_train as train
from workloads.instruction import instruct_evaluate as evaluator
from workloads.instruction import instruct_generation as generation
from workloads.evaluation import mmlu_generation as decoding
from experiments import ifeval_config as ic
from orchestration import mlspace_launch as launch

from experiments.provenance_evaluate import (
    Experiment,
    experiment_configs,
)

ROOT = train.ROOT.parent / 'instruct-provenance-evaluation'
c = train.c


def output_dir(exp):
    return work_path(ROOT / 'runs/out' / exp.name)


def training_arm(exp):
    return next(e for e in train.experiment_configs if e.name == exp.arm)


@lru_cache(maxsize=None)
def checkpoint(arm):
    import torch
    exp = next(e for e in train.experiment_configs if e.name == arm)
    root = train.output_dir(exp)
    marker = json.loads((root / 'trained.json').read_text())
    values, report = train.paired_data()
    expected = train.identity(exp, report, train.make_plan(values))
    if marker['identity'] != expected or not marker['passed'] or marker['checkpoint'] != str(root / 'final.pt'):
        raise RuntimeError('Training completion identity mismatch')
    with train.training_context(exp, values, expected):
        train.common.validate_final(root / 'final.pt', expected['plan'])
    path = root / 'final.pt'
    stat = path.stat()
    return dict(path=str(path), sha256=marker['checkpoint_sha256'], size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, training_signature=train.prep.data.digest(expected))


def protocol():
    from lm_eval.tasks.ifeval import utils, instructions, instructions_registry, instructions_util
    source_root = train.prep.source.SOURCE_ROOT
    gate = json.loads((source_root / 'sanity/generation-protocol.json').read_text())
    original = json.loads((source_root / 'runs/out/qwen3_instruct_original_ar/eval/manifest.json').read_text())
    hashes = {Path(m.__file__).name: train.prep.common.file_hash(m.__file__)
              for m in (utils, instructions, instructions_registry, instructions_util)}
    version = importlib.metadata.version('lm_eval')
    if (gate['generation'] != c.GENERATION or original['ifeval_code'] != hashes or original['lm_eval_version'] != version):
        raise RuntimeError('Scoring or generation protocol differs from source evaluation')
    return dict(ifeval_code=hashes, lm_eval_version=version, generation=c.GENERATION,
                dataset=c.IFEVAL_DATASET, revision=c.IFEVAL_REVISION, seed=c.SEED)


def identity(exp):
    return dict(experiment=exp.__dict__, checkpoint=checkpoint(exp.arm), protocol=protocol(),
                preparation_sha256=train.prep.common.file_hash(train.prep.results() / 'prepared.json'))


def input_path(commit, exp):
    return ROOT / 'submission-inputs' / commit / f'{exp.name}.json'


def require_identity(path, expected):
    if not path.exists() or json.loads(path.read_text()) != expected:
        raise RuntimeError('Evaluation inputs changed or pinned identity unavailable')


def evaluate(exp):
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from langdetect import DetectorFactory
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Evaluation requires exactly one GPU')
    started = time.monotonic()
    ident = identity(exp)
    require_identity(input_path(launch.git('rev-parse', 'HEAD'), exp), ident)
    cp = ident['checkpoint']
    if train.prep.common.file_hash(cp['path']) != cp['sha256']:
        raise RuntimeError('Checkpoint bytes changed')
    root = output_dir(exp) / 'eval'
    meta = root / 'manifest.json'
    if meta.exists():
        require_identity(meta, ident)
    launch.store_json(meta, ident)
    DetectorFactory.seed = c.SEED
    evaluator.ensure_ifeval_resources()
    arm = training_arm(exp)
    model = train.common.load_model(arm.recipe, 'cuda').eval().requires_grad_(False)
    state = torch.load(cp['path'], map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(state['model']); del state
    tokenizer = AutoTokenizer.from_pretrained(c.MODEL, revision=c.REVISION)
    documents = list(load_dataset(c.IFEVAL_DATASET, revision=c.IFEVAL_REVISION, split='train'))
    signature = train.prep.data.digest(ident)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        metrics = evaluator.evaluate_generation(model, tokenizer, documents, 'ifeval', exp.mode,
            c.GENERATION[exp.mode], root, signature, c.SEED, started,
            exp.hours - ic.STARTUP_SECONDS / 3600)
    # Evaluate every model on both fixed paired test corpora, once per model.
    # These are descriptive endpoints; no checkpoint is selected with them.
    crossed = None
    if exp.mode == 'nonthinking':
        if train.prep.common.file_hash(train.prep.results() / 'prepared.json') != ident['preparation_sha256']:
            raise RuntimeError('Held-out preparation changed during generation')
        values, _ = train.paired_data()
        if train.prep.common.file_hash(train.prep.results() / 'prepared.json') != ident['preparation_sha256']:
            raise RuntimeError('Held-out preparation changed while reading')
        crossed = {condition: train.common.nll(model, values[condition]['test'], torch.device('cuda'))
                   for condition in ('corpus', 'teacher')}
        launch.store_json(root / 'crossed-heldout.json', crossed)
    result = dict(identity=ident, metrics=metrics, crossed_heldout=crossed,
                  samples_sha256=train.prep.common.file_hash(root / f'ifeval-{exp.mode}.jsonl'), passed=True)
    launch.store_json(output_dir(exp) / 'evaluated.json', result)


def worker(exp):
    train.prep.bench.child([launch.PYTHON, 'scripts/provenance/provenance_evaluate.py',
                           '--internal', '--filter', exp.name], os.environ.copy(), exp.hours)


def payload(exp, staged):
    body = launch.payload('eval', exp.name, staged, 'd.tarasov', 'mrsndmn')
    body.update(instance_type=launch.INSTANCE_TYPES_BY_NUM_GPUS[1],
                job_desc=f'sps-provenance-ifeval {exp.name} #d.tarasov @mrsndmn')
    body['env_variables']['SPS_EXPERIMENT_SUITE'] = c.SUITE
    body['script'] = f'cd {shlex.quote(str(staged))} && ' + shlex.join([
        str(launch.PYTHON), 'scripts/provenance/provenance_evaluate.py', '--worker', '--filter', exp.name])
    return body


def submit(selected, dry=False):
    commit = launch.git('rev-parse', 'HEAD')
    staged = launch.STAGING_ROOT / commit
    if dry:
        for exp in selected:
            print(json.dumps(payload(exp, staged), indent=2))
        return
    selected = [exp for exp in selected if launch.pending(output_dir(exp))]
    if not selected:
        return
    identities = {exp.name: identity(exp) for exp in selected}
    from mls.manager.job.utils import training_job_api_from_profile
    client, options = training_job_api_from_profile('default')
    launch.disable_submission_retries(client)
    active = {}
    for status in ('Pending', 'Running'):
        response = client.get_list_jobs(options['region'], None, status, 1000, 0)
        if 'jobs' not in response:
            raise RuntimeError('Cannot establish active jobs')
        active.update({launch.normalize(x['job_desc']): x['job_name'] for x in response['jobs']})
    launch.reconcile_for_outputs(client, active, [output_dir(exp) for exp in selected])
    for exp in selected:
        ident = identities[exp.name]
        marker = output_dir(exp) / 'evaluated.json'
        if marker.exists():
            saved = json.loads(marker.read_text())
            samples = output_dir(exp) / 'eval' / f'ifeval-{exp.mode}.jsonl'
            if saved['identity'] != ident or not saved['passed'] or train.prep.common.file_hash(samples) != saved['samples_sha256']:
                raise RuntimeError('Stale evaluation completion')
            continue
        body = payload(exp, staged)
        key = launch.output_identity(body['job_desc'], output_dir(exp), active)
        if key in active:
            job_name = active[key]
            path = ROOT / 'submissions' / f'{job_name}.json'
            if not path.exists():
                raise RuntimeError('Active evaluation has no local receipt')
            receipt = json.loads(path.read_text())
            if receipt['name'] != exp.name or receipt['phase'] != 'eval' or receipt['job_name'] != job_name:
                raise RuntimeError('Active evaluation receipt mismatch')
            require_identity(input_path(receipt['commit'], exp), ident)
            job = dict(name=exp.name, job_name=job_name, reused=True)
        else:
            staged = launch.stage_repo(Path(launch.git('rev-parse', '--show-toplevel')), commit)
            path = ROOT / exp.name / 'budget.json'
            ledger = json.loads(path.read_text()) if path.exists() else dict(attempts=[])
            if sum(x['gpu_hours'] for x in ledger['attempts']) + exp.hours > 2 * exp.hours:
                raise RuntimeError('Evaluation reservation cap exceeded')
            ledger['attempts'].append(dict(gpu_hours=exp.hours, time=time.time()))
            launch.store_json(path, ledger)
            launch.store_json(input_path(commit, exp), ident)
            body = payload(exp, staged); body['region'] = options['region']
            job = launch.submit_payload(client, body, 'eval', exp.name, commit, output_dir=output_dir(exp))
            if job.get("job_name"):
                active[key] = job['job_name']
        print('__MLS_JOBS_JSON__'); print(json.dumps(dict(jobs=[job])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--filter')
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--internal', action='store_true')
    args = parser.parse_args()
    selected = [e for e in experiment_configs if args.filter is None or args.filter == e.name]
    if not selected:
        parser.error('No matching evaluation')
    if args.worker or args.internal:
        if len(selected) != 1:
            parser.error('Worker requires exactly one evaluation')
        return evaluate(selected[0]) if args.internal else worker(selected[0])
    with patch.object(launch, 'ARTIFACTS_ROOT', ROOT), patch.object(launch, 'RUN_ROOT', ROOT / 'runs'):
        if args.dry:
            return submit(selected, True)
        with launch.submission_lock():
            return submit(selected)
