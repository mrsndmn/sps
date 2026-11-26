"""Paired-prompt AR/SPS CE controls after response-provenance preparation.

Primary endpoint is final.pt after one shared prompt pass (complete batches).
Token-normalized CE is unchanged; supervised-token exposure is reported, not
matched. Response source, content and length change together. No test-set or
benchmark score selects this training recipe or its endpoint.
"""
from orchestration.named.publication import work_path, public_path
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shlex
import sys
import time
from unittest.mock import patch

from orchestration.source_compatibility import source_hash
from workloads.provenance import provenance_prepare as prep
from experiments import instruct_config as c
from workloads.instruction import instruct_common as common
from workloads.instruction import instruct_train as trainer
from orchestration import mlspace_launch as launch

from experiments.provenance_train import (
    Experiment,
    experiment_configs,
)

ROOT = prep.ROOT.parent / 'instruct-provenance-controls'


def output_dir(exp):
    return work_path(ROOT / 'runs/out' / exp.name)


def paired_data():
    if not prep.completed():
        raise RuntimeError('Paired preparation is not complete')
    report = json.loads((prep.results() / 'prepared.json').read_text())
    values = {condition: {split: [json.loads(line) for line in
                  Path(report['splits'][split]['conditions'][condition]['path']).read_text().splitlines()]
              for split in prep.SPLITS} for condition in ('corpus', 'teacher')}
    validate_pairs(values, report)
    return values, report


def validate_pairs(values, report):
    seen = set()
    groups = {}
    for split in prep.SPLITS:
        a, b = values['corpus'][split], values['teacher'][split]
        if len(a) != len(b) or not a:
            raise RuntimeError('Unpaired split sizes')
        for x, y in zip(a, b):
            if any(x[k] != y[k] for k in ('source_id', 'group_id', 'mode')) or x['messages'][:-1] != y['messages'][:-1]:
                raise RuntimeError('Prompt identity mismatch')
            if x['mode'] == 'thinking' and x != y:
                raise RuntimeError('Thinking row changed')
            if x['source_id'] in seen:
                raise RuntimeError('Duplicate paired source')
            seen.add(x['source_id'])
            group = x['group_id']
            if group in groups and groups[group] != split:
                raise RuntimeError('Group crosses splits')
            groups[group] = split
        if prep.data.digest([r['source_id'] for r in a]) != report['splits'][split]['paired_source_ids_sha256']:
            raise RuntimeError('Paired order digest mismatch')
        for condition in ('corpus', 'teacher'):
            if prep.summarize(values[condition][split]) != report['splits'][split]['conditions'][condition]['by_mode']:
                raise RuntimeError('Paired token accounting mismatch')


def ordered_train(values):
    return sorted(values, key=lambda r: prep.data.digest([c.SEED, r['source_id']]))


def make_plan(values):
    steps = min(len(values['corpus']['train']) // c.GLOBAL_BATCH, c.MAX_STEPS)
    transition = int(steps * c.STAGE_FRACTION)
    if not 0 < transition < steps:
        raise RuntimeError('Insufficient paired training data')
    count = steps * c.GLOBAL_BATCH
    used = {condition: ordered_train(values[condition]['train'])[:count] for condition in ('corpus', 'teacher')}
    if [r['source_id'] for r in used['corpus']] != [r['source_id'] for r in used['teacher']]:
        raise RuntimeError('Training order differs')
    return dict(steps=steps, stage_steps=transition, global_batch=c.GLOBAL_BATCH,
                selected_examples=count, dropped_tail=len(values['corpus']['train'])-count,
                order_sha256=prep.data.digest([r['source_id'] for r in used['corpus']]),
                exposure={condition: dict(all=prep.summarize(rows),
                    initial_phase=prep.summarize(rows[:transition*c.GLOBAL_BATCH]),
                    full_phase=prep.summarize(rows[transition*c.GLOBAL_BATCH:]))
                          for condition, rows in used.items()},
                endpoint='final.pt', scope='Matched prompts/order/steps; token exposure not matched; own-condition validation NLL is monitoring only.')


def identity(exp, report, plan):
    files = [Path(__file__), Path(trainer.__file__), Path(common.__file__), Path(c.__file__),
             Path(prep.__file__), Path('src/modeling/models/sps/pretrained.py')]
    return dict(experiment=exp.__dict__, preparation=report, plan=plan,
                code={p.name: source_hash(p) for p in files})


@contextmanager
def training_context(exp, values, ident):
    signature = prep.data.digest(ident)
    e = trainer.e
    def rows(split):
        records = values[exp.condition][split]
        return ordered_train(records) if split == 'train' else records
    with patch.object(e, 'output_dir', output_dir), patch.object(e, 'RUN_ROOT', ROOT / 'runs'), \
         patch.object(e, 'collect_experiments', lambda name: [exp] if name == exp.name else []), \
         patch.object(e, 'design_signature', lambda: signature), \
         patch.object(trainer, 'rows', rows), patch.object(trainer, 'read_plan', lambda: ident['plan']):
        yield


def train_worker(exp):
    values, report = paired_data()
    ident = identity(exp, report, make_plan(values))
    saved = json.loads((output_dir(exp) / 'identity.json').read_text())
    if ident != saved:
        raise RuntimeError('Training inputs changed')
    with training_context(exp, values, ident), patch.object(sys, 'argv', [__file__, '--filter', exp.name]):
        trainer.main()


def verify_submitted_identity(exp, ident):
    path = submission_input_path(launch.git('rev-parse', 'HEAD'), exp)
    if not path.exists() or json.loads(path.read_text()) != ident:
        raise RuntimeError('Queued training inputs differ from submission or provenance is missing')


def worker(exp):
    values, report = paired_data()
    ident = identity(exp, report, make_plan(values))
    verify_submitted_identity(exp, ident)
    root = output_dir(exp)
    path = root / 'identity.json'
    if path.exists() and json.loads(path.read_text()) != ident:
        raise RuntimeError('Training output uses other inputs')
    launch.store_json(path, ident)
    import os
    prep.bench.child([launch.PYTHON, '-m', 'torch.distributed.run', '--standalone',
                      f'--nproc_per_node={exp.num_gpus}', 'scripts/provenance/provenance_train.py',
                      '--train-worker', '--arm', exp.name], os.environ.copy(), exp.hours-300/3600)
    with training_context(exp, values, ident):
        common.validate_final(root / 'final.pt', ident['plan'])
    launch.store_json(root / 'trained.json', dict(identity=ident, checkpoint=str(public_path(root / 'final.pt')),
                      checkpoint_sha256=prep.common.file_hash(root / 'final.pt'), passed=True))


def payload(exp, staged):
    body = launch.payload('train', exp.name, staged, 'd.tarasov', 'mrsndmn')
    body.update(instance_type=launch.INSTANCE_TYPES_BY_NUM_GPUS[exp.num_gpus],
                job_desc=f'sps-provenance-controls train {exp.name} #d.tarasov @mrsndmn')
    body['env_variables']['SPS_EXPERIMENT_SUITE'] = c.SUITE
    body['script'] = f'cd {shlex.quote(str(staged))} && ' + shlex.join([
        str(launch.PYTHON), 'scripts/provenance/provenance_train.py', '--worker', '--arm', exp.name])
    return body


def submission_input_path(commit, exp):
    return ROOT / 'submission-inputs' / commit / f'{exp.name}.json'


def verify_active_identity(exp, job_name, ident):
    receipt_path = ROOT / 'submissions' / f'{job_name}.json'
    if not receipt_path.exists():
        raise RuntimeError('Active training job has no local provenance receipt')
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get('job_name') != job_name or receipt.get('name') != exp.name
            or receipt.get('phase') != 'train'):
        raise RuntimeError('Active training receipt mismatch')
    path = submission_input_path(receipt['commit'], exp)
    if not path.exists() or json.loads(path.read_text()) != ident:
        raise RuntimeError('Active training job inputs differ or are unavailable')


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
    values, report = paired_data()
    plan = make_plan(values)
    launch.store_json(ROOT / 'plan.json', dict(plan=plan, preparation_sha256=prep.common.file_hash(prep.results() / 'prepared.json')))
    from mls.manager.job.utils import training_job_api_from_profile
    client, options = training_job_api_from_profile('default')
    launch.disable_submission_retries(client)
    active = {}
    for status in ('Pending', 'Running'):
        response = client.get_list_jobs(options['region'], None, status, 1000, 0)
        if 'jobs' not in response:
            raise RuntimeError('Cannot establish active jobs')
        active.update({launch.normalize(x['job_desc']): x['job_name'] for x in response['jobs']})
    launch.reconcile_receipts(client, active)
    launch.reconcile_intents(active)
    jobs = []
    for exp in selected:
        ident = identity(exp, report, plan)
        marker = output_dir(exp) / 'trained.json'
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved['identity'] != ident or not saved['passed'] or prep.common.file_hash(saved['checkpoint']) != saved['checkpoint_sha256']:
                raise RuntimeError('Stale completed control')
            continue
        body = payload(exp, staged)
        key = launch.output_identity(body['job_desc'], output_dir(exp), active)
        if key in active:
            verify_active_identity(exp, active[key], ident)
            job = dict(name=exp.name, job_name=active[key], reused=True)
        else:
            staged = launch.stage_repo(Path(launch.git('rev-parse', '--show-toplevel')), commit)
            body = payload(exp, staged)
            body['region'] = options['region']
            path = ROOT / exp.name / 'budget.json'
            ledger = json.loads(path.read_text()) if path.exists() else dict(attempts=[])
            cost = exp.num_gpus * exp.hours
            if sum(x['gpu_hours'] for x in ledger['attempts']) + cost > 2 * cost:
                raise RuntimeError('Training reservation cap exceeded')
            ledger['attempts'].append(dict(gpu_hours=cost, time=time.time()))
            launch.store_json(path, ledger)
            # Persist before API send so lost-response intent reconciliation can
            # recover both the UUID receipt and its exact input identity.
            launch.store_json(submission_input_path(commit, exp), ident)
            job = launch.submit_payload(client, body, 'train', exp.name, commit, output_dir=output_dir(exp))
            if job.get("job_name"):
                active[key] = job['job_name']
        jobs.append(job)
        # Emit each receipt immediately so partial submission is observable.
        print('__MLS_JOBS_JSON__'); print(json.dumps(dict(jobs=[job])), flush=True)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['all']+[e.name for e in experiment_configs], default='all')
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--train-worker', action='store_true')
    args = parser.parse_args()
    selected = [e for e in experiment_configs if args.arm in ('all', e.name)]
    if args.worker or args.train_worker:
        if len(selected) != 1:
            parser.error('Worker requires exactly one arm')
        return train_worker(selected[0]) if args.train_worker else worker(selected[0])
    with patch.object(launch, 'ARTIFACTS_ROOT', ROOT), patch.object(launch, 'RUN_ROOT', ROOT / 'runs'):
        if args.dry:
            return submit(selected, True)
        with launch.submission_lock():
            return submit(selected)
