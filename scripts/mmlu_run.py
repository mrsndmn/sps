"""Resumable single-GPU MMLU reasoning evaluation of pinned final checkpoints."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import experiments as e
from modeling.models.sps.pretrained import PretrainedSPS
from scripts import mmlu_config as c
from scripts.mmlu_results import recover_samples
from scripts.mmlu_generation import generate, prompt, parse_answer


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def load_model(exp, device):
    base = AutoModelForCausalLM.from_pretrained(exp.model, revision=exp.revision,
             torch_dtype=torch.float32, attn_implementation='sdpa')
    model = PretrainedSPS(base, sps=exp.variant != 'base')
    if exp.variant != 'base':
        checkpoint = torch.load(e.COMMON_ROOT / exp.suite / 'runs/out' / exp.arm / 'final.pt',
                                map_location='cpu', weights_only=False, mmap=True)
        if (checkpoint.get('design_signature') != exp.signature or
                checkpoint.get('source') != exp.model or checkpoint.get('revision') != exp.revision):
            raise ValueError(f'Checkpoint provenance mismatch: {exp.name}')
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
    return model.to(device).eval().requires_grad_(False)


def evaluate(exp, limit=None):
    if e.SUITE != 'mmlu-generation':
        raise ValueError('Set SPS_EXPERIMENT_SUITE=mmlu-generation')
    torch.manual_seed(c.SEED)
    started = time.monotonic()
    root = e.output_dir(exp) / ('smoke' if limit else 'full')
    root.mkdir(parents=True, exist_ok=True)
    signature = e.design_signature()
    manifest = dict(design_signature=signature, experiment=exp.__dict__, limit=limit)
    meta = root / 'manifest.json'
    if meta.exists() and json.loads(meta.read_text()) != manifest:
        raise ValueError('Resume manifest mismatch')
    write_json(meta, manifest)
    dataset = load_dataset(c.DATASET, 'all', revision=c.REVISION, split='test')
    rows = list(dataset.select(range(min(limit, len(dataset))))) if limit else list(dataset)
    path = root / 'samples.jsonl'
    saved = recover_samples(path)
    for index, item in enumerate(saved):
        digest = hashlib.sha256(json.dumps(rows[index], sort_keys=True).encode()).hexdigest()
        if item['index'] != index or item['doc_hash'] != digest:
            raise ValueError('Resume dataset/order mismatch')
    model = load_model(exp, 'cuda')
    tokenizer = AutoTokenizer.from_pretrained(exp.model, revision=exp.revision)
    with path.open('a') as stream, torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for index in range(len(saved), len(rows)):
            if time.monotonic() - started > c.MAX_HOURS * 3600:
                raise TimeoutError('Evaluation deadline reached; resume the same experiment')
            row = rows[index]
            try:
                text, tokens, stop = generate(model, tokenizer, prompt(row), c.MAX_NEW_TOKENS)
            except ValueError as error:
                if 'exceeds native context' not in str(error):
                    raise
                text, tokens, stop = '', [], 'context_overflow'
            answer = parse_answer(text)
            item = dict(index=index, doc_hash=hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
                        subject=row['subject'], target='ABCD'[row['answer']], answer=answer,
                        correct=answer == 'ABCD'[row['answer']], generation=text, tokens=tokens, stop=stop)
            stream.write(json.dumps(item) + '\n'); stream.flush(); os.fsync(stream.fileno())
            saved.append(item)
            if (index + 1) % 10 == 0:
                print(json.dumps(dict(completed=index+1, total=len(rows), seconds=time.monotonic()-started)), flush=True)
    subjects = sorted({item['subject'] for item in saved})
    per_subject = {subject: sum(x['correct'] for x in saved if x['subject'] == subject) /
                   sum(x['subject'] == subject for x in saved) for subject in subjects}
    result = dict(manifest, count=len(saved), accuracy=sum(x['correct'] for x in saved)/len(saved),
                  macro_accuracy=sum(per_subject.values())/len(per_subject), per_subject=per_subject,
                  invalid_rate=sum(x['answer'] is None for x in saved)/len(saved),
                  length_rate=sum(x['stop'] == 'length' for x in saved)/len(saved),
                  context_overflow_rate=sum(x['stop'] == 'context_overflow' for x in saved)/len(saved))
    write_json(root / 'results.json', result)
    if not limit:
        write_json(e.output_dir(exp) / 'evaluated.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['eval'])
    parser.add_argument('--filter', required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    matches = [x for x in c.experiment_configs if x.name == args.filter]
    if len(matches) != 1 or (args.limit is not None and args.limit <= 0):
        parser.error('Use one exact experiment name and a positive limit')
    evaluate(matches[0], args.limit)

if __name__ == '__main__':
    main()
