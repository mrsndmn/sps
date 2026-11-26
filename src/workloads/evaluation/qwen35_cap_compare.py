"""Paired fresh thinking generations on the archived teacher-cap cohort."""
from orchestration.named.publication import worker_output
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess

from experiments.qwen35_cap_compare import OUTPUT_PARENT
from experiments import instruct_config as c
from data import instruct_data as data
from workloads.serving import serving_runtime as runtime
from workloads.evaluation import teacher_cap_probe as pilot
from orchestration import mlspace_launch as launch
from experiments.qwen35_cap_compare import BATCH, CAPS, RUNTIME_ENV, experiment_configs

ROOT = pilot.ROOT.parent / 'qwen35-unloop-cap-compare'
PROMPTS = ROOT / 'cohort.json'


def prepare():
    rows = []
    for domain, size, row, archived in pilot.sample(pilot.Probe()):
        key = row['source_id']
        prev = json.loads((pilot.ROOT/pilot.PILOT_HASH/'samples'/f'{key}.json').read_bytes())
        messages = row['messages'][:-1]
        if prev['identity']['input_hash'] != data.digest([messages, archived['generated_ids']]):
            raise ValueError('Previous probe input differs')
        rows.append(dict(id=key, domain=domain, population=size, messages=messages,
                         seed=archived['seed']))
    assert len({r['id'] for r in rows}) == len(rows) == 109
    if PROMPTS.exists() and json.loads(PROMPTS.read_bytes()) != rows:
        raise ValueError('Cohort mismatch')
    data.atomic_json(PROMPTS, rows)


def folder(arm):
    return worker_output(ROOT.joinpath(*OUTPUT_PARENT, arm))


def weights_path(arm):
    experiment = experiment_configs[arm]
    repo, revision = experiment.model, experiment.revision
    return launch.HF_HOME / 'hub' / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision


def generate(arm):
    os.environ.update(RUNTIME_ENV)
    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    assert torch.cuda.device_count() == 1
    rows = json.loads(PROMPTS.read_bytes())
    out_dir = folder(arm)
    model_path = str(weights_path(arm))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = {}
    for row in rows:
        prompt = tokenizer.apply_chat_template(row['messages'], tokenize=False, enable_thinking=True,
                                               add_generation_prompt=True)
        if not prompt.endswith('<think>\n'):
            raise ValueError('Missing thinking prefix')
        prompts[row['id']] = tokenizer.encode(prompt, add_special_tokens=False)
    end_ids = {tokenizer.eos_token_id,
               json.loads((Path(model_path)/'config.json').read_bytes())['text_config']['eos_token_id']}
    needed = [r for r in rows if not (out_dir/(r['id']+'.json')).exists()]
    if not needed:
        return
    engine = LLM(model=model_path, dtype='bfloat16', language_model_only=True, enforce_eager=True,
                 generation_config='vllm', max_num_seqs=BATCH, gpu_memory_utilization=.82,
                 max_model_len=max(map(len,prompts.values()))+CAPS[-1])
    for i in range(0, len(needed), BATCH):
        batch = needed[i:i+BATCH]
        params = [SamplingParams(max_tokens=CAPS[-1], temperature=c.TEACHER_TEMPERATURE,
                  top_k=c.TEACHER_TOP_K, top_p=c.TEACHER_TOP_P, seed=r['seed'],
                  stop_token_ids=sorted(end_ids)) for r in batch]
        outputs = engine.generate([dict(prompt_token_ids=prompts[r['id']]) for r in batch], params)
        for r, response in zip(batch, outputs, strict=True):
            out, = response.outputs
            ids = list(out.token_ids)
            assert list(response.prompt_token_ids)==prompts[r['id']] and out.finish_reason in ('stop','length')
            # Qwen3.5 puts the opening tag inside the prompt. Preserve generated closing tags.
            clean = ids[:-1] if ids and ids[-1] in end_ids else ids
            text = tokenizer.decode(clean, skip_special_tokens=False)
            data.atomic_json(out_dir/(r['id']+'.json'),dict(id=r['id'],revision=experiment_configs[arm].revision,
                seed=r['seed'],prompt_hash=data.digest(prompts[r['id']]),ids=ids,text=text,stop=out.finish_reason))


def tail_loop(ids):
    tail = ids[-256:]
    return any(tail[i] == tail[i+k] == tail[i+2*k] and
               tail[i:i+k] == tail[i+k:i+2*k] == tail[i+2*k:i+3*k]
               for k in range(8,65) for i in range(len(tail)-3*k+1))


def completed_answer(text, stop):
    if stop != 'stop':
        return None
    answer = data.parse_thinking('<think>\n'+text)
    return answer if answer and answer['content'] else None


def summarize(arm):
    from transformers import AutoTokenizer
    rows = json.loads(PROMPTS.read_bytes())
    student = AutoTokenizer.from_pretrained(c.MODEL, revision=c.REVISION)
    counts = Counter(r['domain'] for r in rows)
    records = []
    totals = {cap: Counter() for cap in CAPS}
    for r in rows:
        item = json.loads((folder(arm)/(r['id']+'.json')).read_bytes())
        assert (item['id'], item['seed'], item['revision']) == (r['id'], r['seed'], experiment_configs[arm].revision)
        ids = item['ids']
        answer = completed_answer(item['text'], item['stop'])
        fits = False
        if answer:
            try:
                data.encode_example(student,r['messages']+[answer],'thinking',r['id'],c.CONTEXT)
                fits = True
            except ValueError as e:
                if 'exceeds context' not in str(e):
                    raise
        records.append(dict(id=r['id'],prompt_hash=item['prompt_hash'],domain=r['domain'],tokens=len(ids),
                            stop=item['stop'],complete=bool(answer),fits=fits))
        for cap, acc in totals.items():
            w = r['population']/counts[r['domain']]
            for key, v in dict(weight=1, complete=bool(answer) and len(ids)<=cap, fits=fits and len(ids)<=cap,
                               capped=len(ids)>cap or (len(ids)==cap and item['stop']=='length'),
                               tail_loop=tail_loop(ids[:cap]), cost=min(len(ids),cap)).items():
                acc[key] += w*v
    curve = {cap:{k:v/acc['weight'] for k,v in acc.items() if k!='weight'} for cap,acc in totals.items()}
    data.atomic_json(folder(arm)/'report.json',dict(records=records,curve=curve))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','worker','generate'])
    parser.add_argument('--arm', choices=experiment_configs)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    elif args.arm is None:
        parser.error('--arm required')
    elif args.phase == 'generate':
        generate(args.arm)
    else:
        subprocess.run([runtime.RUNTIME/'bin/python','scripts/evaluation/qwen35_cap_compare.py','generate',
                        '--arm',args.arm],env=runtime.runtime_environment(),check=True)
        summarize(args.arm)
