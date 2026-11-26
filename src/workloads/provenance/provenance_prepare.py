"""Prepare paired corpus/teacher non-thinking responses on the same source prompts.

Thinking rows are copied verbatim. Rejecting a generated response removes the
corresponding prompt from BOTH conditions. No benchmark score selects responses.
This stage reports actual token budgets before training configurations are fixed.
"""
from orchestration.named.publication import work_path, public_path
from experiments.launch import named_run, run_filter
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shlex
import time
from unittest.mock import patch

from experiments import instruct_config as c
from data import instruct_data as data
from orchestration import mlspace_launch as launch
from workloads.serving import serving_benchmark as bench
from workloads.serving import vllm_sps_common as common
from experiments import vllm_sps_config as source

from experiments.provenance_prepare import family_config as provenance_prepare_settings
from experiments.provenance_prepare import Experiment, experiment_configs

REFERENCE_JOB = named_run(experiment_configs, Experiment.name)
SPLITS = provenance_prepare_settings.SPLITS
GENERATION = provenance_prepare_settings.GENERATION
MIN_PAIRED_FRACTION = provenance_prepare_settings.MIN_PAIRED_FRACTION

ROOT = source.family_config.ROOT.parent/'instruct-response-provenance'


def prepared_dir(job):
    return ROOT / (job.directory or job.name)


def results(job):
    path = work_path(prepared_dir(job))
    if job.directory == "prepared" and path == prepared_dir(job) and not path.exists() and (ROOT / "prepared.json").is_file():
        return ROOT
    return path


def signature():
    return data.digest(dict(generation=GENERATION, model=c.family_config.MODEL, revision=c.family_config.REVISION))


def source_rows():
    rows = {split: common.data(split) for split in SPLITS}
    seen = set()
    groups = {}
    for split, records in rows.items():
        for row in records:
            if row['source_id'] in seen:
                raise RuntimeError('Duplicate source prompt')
            seen.add(row['source_id'])
            if row['group_id'] in groups and groups[row['group_id']] != split:
                raise RuntimeError('Prompt group crosses data splits')
            groups[row['group_id']] = split
    return rows


def identity():
    manifest = json.loads((source.family_config.SOURCE_ROOT/'runs/data/manifest.json').read_text())
    return dict(design_signature=signature(), source_manifest=manifest,
                model=c.family_config.MODEL, revision=c.family_config.REVISION, generation=GENERATION)


def record_path(row, job):
    name = row['source_id']
    if len(name) != 64 or any(x not in '0123456789abcdef' for x in name):
        raise ValueError('Invalid source identifier')
    return results(job)/'samples'/f'{name}.json'


def validate_record(record, row, split, expected):
    if (record['source_id'] != row['source_id'] or record['source_row_sha256'] != data.digest(row) or record['split'] != split):
        raise RuntimeError('Cached teacher response identity mismatch')
    if ('row' in record) == ('rejected' in record):
        raise RuntimeError('Teacher response must be accepted or rejected')
    if 'row' in record:
        new = record['row']
        if (new['messages'][:-1] != row['messages'][:-1] or new['mode'] != row['mode']
                or new['source_id'] != row['source_id'] or new['group_id'] != row['group_id']):
            raise RuntimeError('Teacher response changed prompt identity')


def generate_worker(job):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != job.num_gpus:
        raise RuntimeError('Unexpected preparation allocation')
    device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
    torch.cuda.set_device(device)
    expected = signature()
    records = [(split,row) for split,rows in source_rows().items() for row in rows if row['mode']=='nonthinking']
    tokenizer = AutoTokenizer.from_pretrained(c.family_config.MODEL,revision=c.family_config.REVISION)
    teacher = None
    for i,(split,row) in enumerate(records):
        if i % world != rank:
            continue
        path = record_path(row, job=job)
        if path.exists():
            validate_record(json.loads(path.read_text()),row,split,expected)
            continue
        if teacher is None:
            teacher = AutoModelForCausalLM.from_pretrained(c.family_config.MODEL,revision=c.family_config.REVISION,
                torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval().requires_grad_(False)
        prompt = tokenizer.apply_chat_template(row['messages'][:-1],tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_tensors='pt').to(device)
        budget = min(GENERATION['max_new_tokens'], c.family_config.CONTEXT-prompt.shape[1])
        record = dict(design_signature=expected,source_id=row['source_id'],split=split,
                      source_row_sha256=data.digest(row))
        if budget < 1:
            record['rejected']='no_generation_room'
            data.atomic_json(path,record); continue
        seed=(c.family_config.SEED+int(row['source_id'][:8],16)) % 2**31
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            generated=teacher.generate(prompt,attention_mask=torch.ones_like(prompt),do_sample=True,
                max_new_tokens=budget,temperature=GENERATION['temperature'],top_p=GENERATION['top_p'],
                top_k=GENERATION['top_k'],pad_token_id=tokenizer.eos_token_id)
        tokens=generated[0,prompt.shape[1]:].tolist()
        eos=teacher.generation_config.eos_token_id
        eos=eos if isinstance(eos,list) else [eos]
        text=tokenizer.decode(tokens,skip_special_tokens=True).strip()
        special={tokenizer.convert_tokens_to_ids(x) for x in ('<think>','</think>')}
        record.update(seed=seed,generated_ids=tokens,generated_text=text,max_new_tokens=budget,
                      prompt_tokens=prompt.shape[1])
        if not tokens or tokens[-1] not in eos:
            record['rejected']='incomplete_response'
        elif not text or special.intersection(tokens) or '<think>' in text or '</think>' in text:
            record['rejected']='empty_or_malformed_nonthinking'
        else:
            messages=row['messages'][:-1]+[dict(role='assistant',content=text)]
            try:
                record['row']=data.encode_example(tokenizer,messages,'nonthinking',row['source_id'],c.family_config.CONTEXT)
            except ValueError as error:
                record['rejected']=str(error)
        validate_record(record,row,split,expected)
        data.atomic_json(path,record)
        print(json.dumps(dict(rank=rank,source_id=row['source_id'],accepted='row' in record)),flush=True)


def pair_rows(rows, samples, split, expected):
    original, teacher = [],[]
    rejected=Counter()
    for row in rows:
        if row['mode']=='thinking':
            original.append(row);teacher.append(row);continue
        record=samples[row['source_id']]
        validate_record(record,row,split,expected)
        if 'rejected' in record:
            rejected[record['rejected']]+=1;continue
        original.append(row);teacher.append(record['row'])
    counts={mode:sum(r['mode']==mode for r in original) for mode in c.family_config.GENERATION}
    source_count=sum(r['mode']=='nonthinking' for r in rows)
    if not all(counts.values()) or counts['nonthinking'] < MIN_PAIRED_FRACTION*source_count:
        raise RuntimeError(f'Insufficient paired data in {split}: {counts}; rejected={dict(rejected)}')
    return original,teacher,dict(rejected)


def summarize(rows):
    result={}
    for mode in c.family_config.GENERATION:
        selected=[r for r in rows if r['mode']==mode]
        tokens=sorted(sum(r['loss_mask'][1:]) for r in selected)
        result[mode]=dict(examples=len(selected),supervised_tokens=sum(tokens),
            input_tokens=sum(len(r['input_ids']) for r in selected),
            target_length_min=tokens[0],target_length_median=tokens[len(tokens)//2],target_length_max=tokens[-1])
    return result


def finalize(job):
    expected=signature()
    report=dict(identity=identity(),splits={},
        scope='Paired accepted prompts; thinking unchanged. Response length and supervised tokens are measured, not assumed equal. Training is a separate next stage.')
    for split,rows in source_rows().items():
        samples={r['source_id']:json.loads(record_path(r, job=job).read_text()) for r in rows if r['mode']=='nonthinking'}
        original,teacher,rejected=pair_rows(rows,samples,split,expected)
        group=dict(rejected=rejected,conditions={})
        for condition,values in (('corpus',original),('teacher',teacher)):
            path=results(job)/'data'/condition/f'{split}.jsonl'
            path.parent.mkdir(parents=True,exist_ok=True)
            data.write_jsonl(path,values)
            group['conditions'][condition]=dict(path=str(public_path(path)),sha256=common.file_hash(path),by_mode=summarize(values))
        group['paired_source_ids_sha256']=data.digest([r['source_id'] for r in original])
        report['splits'][split]=group
    data.atomic_json(results(job)/'prepared.json',report)
    print(json.dumps(report,indent=2),flush=True)


def worker(job):
    current=identity()
    meta=results(job)/'identity.json'
    if meta.exists() and json.loads(meta.read_text()) != current:
        raise RuntimeError('Preparation inputs changed')
    data.atomic_json(meta,current)
    source_rows()
    bench.child([launch.PYTHON,'-m','torch.distributed.run','--standalone',
                 f'--nproc_per_node={job.num_gpus}','scripts/provenance/provenance_prepare.py','--generate-worker','--filter',job.name],
                os.environ.copy(),job.hours-300/3600)
    finalize(job=job)


def completed(job):
    marker=results(job)/'prepared.json'
    if not marker.exists():return False
    report=json.loads(marker.read_text())
    if report['identity'] != identity():raise RuntimeError('Stale paired preparation')
    source_rows()
    for split in SPLITS:
        for condition in ('corpus','teacher'):
            value=report['splits'][split]['conditions'][condition]
            if common.file_hash(value['path']) != value['sha256']:
                raise RuntimeError('Paired data checksum mismatch')
    return True


def payload(exp,staged):
    body=launch.payload('train',exp.name,staged,'d.tarasov','mrsndmn')
    body.update(instance_type=launch.INSTANCE_TYPES_BY_NUM_GPUS[exp.num_gpus],
                job_desc=f'sps-response-provenance prepare {exp.name} #d.tarasov @mrsndmn')
    body['script']=f'cd {shlex.quote(str(staged))} && '+shlex.join([str(launch.PYTHON),'scripts/provenance/provenance_prepare.py','--worker','--filter',exp.name])
    return body


def submit(job, dry=False):
    commit=launch.git('rev-parse','HEAD');staged=launch.STAGING_ROOT/commit
    if dry:
        launch.print_preview(payload(job,staged), prepared_dir(job), staged, "prepare");return
    if not launch.pending(prepared_dir(job)):
        return
    if completed(job=job):print('Already completed');return
    source_rows()
    def reserve():
        path=ROOT/'budget.json';ledger=json.loads(path.read_text()) if path.exists() else {'attempts':[]}
        cost=job.hours*job.num_gpus
        if sum(x['gpu_hours'] for x in ledger['attempts'])+cost > 2*cost:
            raise RuntimeError('Preparation reservation cap exceeded')
        ledger['attempts'].append(dict(gpu_hours=cost,time=time.time()));launch.store_json(path,ledger)

    receipt = launch.SubmissionBatch(commit).submit(payload(job, staged), "prepare", job.name, prepared_dir(job), before_send=reserve)
    print("__MLS_JOBS_JSON__"); print(json.dumps({"jobs": [receipt]}), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--generate-worker',action='store_true')
    jobs = run_filter(parser, experiment_configs)
    args=parser.parse_args()
    job = jobs[args.filter]
    if args.generate_worker:return generate_worker(job=job)
    if args.worker:return worker(job=job)
    with patch.object(launch,'ARTIFACTS_ROOT',ROOT),patch.object(launch,'RUN_ROOT',ROOT/'runs'):
        if args.dry:return submit(job, True)
        with launch.submission_lock():return submit(job)
