"""Resumable final-checkpoint instruct evaluation; one GPU per arm."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path
import torch
from scripts.instruct_generation import generate, document_seed, mmlu_messages, parse_answer
from scripts.mmlu_results import recover_samples


def ensure_ifeval_resources():
    """Keep NLTK resources on shared storage, including fresh job containers."""
    import experiments as e
    import nltk
    root = e.COMMON_ROOT / 'nltk_data'
    root.mkdir(parents=True, exist_ok=True)
    if str(root) not in nltk.data.path:
        nltk.data.path.insert(0, str(root))
    try:
        nltk.data.find('tokenizers/punkt_tab/english/')
    except LookupError:
        nltk.download('punkt_tab', download_dir=str(root), quiet=True, raise_on_error=True)
    from lm_eval.tasks.ifeval.utils import process_results
    process_results(dict(key=0, prompt='Say hello', instruction_id_list=['keywords:existence'],
                         kwargs=[{'keywords': ['hello']}]), ['hello'])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)))
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_saved(saved, rows):
    seen = set()
    for item in saved:
        i = item['index']
        if i in seen or not 0 <= i < len(rows) or item['doc_hash'] != digest(rows[i]):
            raise ValueError('Duplicate or mismatched instruct evaluation sample')
        seen.add(i)
    return seen


def aggregate(samples, task):
    result = dict(count=len(samples), length_rate=sum(x['stop'] == 'length' for x in samples)/len(samples),
                  malformed_thinking_rate=sum(not x['thinking_closed'] for x in samples)/len(samples))
    if task == 'mmlu':
        from scripts.report_mmlu_stem import STEM_SUBJECTS
        stem = [x for x in samples if x['subject'] in STEM_SUBJECTS]
        result.update(accuracy=sum(x['correct'] for x in samples)/len(samples),
                      invalid_rate=sum(x['answer'] is None for x in samples)/len(samples),
                      stem_count=len(stem), stem_accuracy=sum(x['correct'] for x in stem)/len(stem) if stem else None)
    else:
        for key in ('prompt_level_strict_acc', 'prompt_level_loose_acc'):
            result[key] = sum(x['metrics'][key] for x in samples)/len(samples)
        for key in ('inst_level_strict_acc', 'inst_level_loose_acc'):
            values = [v for x in samples for v in x['metrics'][key]]
            result[key] = sum(values)/len(values)
    return result


def evaluate_generation(model, tokenizer, rows, task, mode, settings, destination, signature, seed, started, max_hours):
    from lm_eval.tasks.ifeval.utils import process_results
    path = destination / f'{task}-{mode}.jsonl'
    saved = recover_samples(path)
    seen = validate_saved(saved, rows)
    with path.open('a') as stream:
        for index, row in enumerate(rows):
            if index in seen:
                continue
            if time.monotonic() - started > max_hours * 3600:
                raise TimeoutError('Instruct evaluation deadline reached; saved answers can be resumed')
            messages = mmlu_messages(row) if task == 'mmlu' else [{'role': 'user', 'content': row['prompt']}]
            doc_seed = document_seed(seed, task, mode, index)
            result = generate(model, tokenizer, messages, mode == 'thinking', settings, doc_seed)
            item = dict(index=index, doc_hash=digest(row), seed=doc_seed, **result)
            if task == 'mmlu':
                answer = parse_answer(result['final_text'])
                target = 'ABCD'[row['answer']]
                item.update(answer=answer, target=target, correct=answer == target, subject=row['subject'])
            else:
                random.seed(doc_seed)
                item['metrics'] = process_results(row, [result['final_text']])
            stream.write(json.dumps(item) + '\n'); stream.flush(); os.fsync(stream.fileno())
            saved.append(item)
            if len(saved) % 10 == 0:
                print(json.dumps(dict(task=task, mode=mode, completed=len(saved), total=len(rows))), flush=True)
    result = dict(design_signature=signature, **aggregate(saved, task))
    write_json(destination / f'{task}-{mode}.json', result)
    return result


def likelihood(model, tokenizer, root, limit):
    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator
    from scripts import instruct_config as c
    class Adapter(HFLM):
        def _model_call(self, inps, attn_mask=None, labels=None):
            if attn_mask is not None or labels is not None:
                raise ValueError('Only causal likelihood scoring is supported')
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                return model(inps).logits
    adapter = Adapter(pretrained=model.backbone, tokenizer=tokenizer, backend='causal',
                      batch_size=c.EVAL_BATCH, max_length=c.LIKELIHOOD_CONTEXT,
                      device='cuda', logits_cache=False)
    for task in c.LIKELIHOOD_TASKS:
        path = root / f'{task}.json'
        if path.exists():
            continue
        source = c.LIKELIHOOD_DATASETS[task]
        task_config = dict(task=task, dataset_path=source['repo'],
                           dataset_kwargs=dict(revision=source['revision'],
                                               trust_remote_code=task == 'lambada_openai'))
        if 'config' in source:
            task_config['dataset_name'] = source['config']
        result = evaluator.simple_evaluate(model=adapter, tasks=[task_config], num_fewshot=0,
                 batch_size=c.EVAL_BATCH, log_samples=True, limit=limit, random_seed=c.SEED,
                 numpy_random_seed=c.SEED, torch_random_seed=c.SEED, fewshot_random_seed=c.SEED)
        write_json(path, result)


def evaluate(exp, limit=None):
    import experiments as e
    from scripts import instruct_config as c
    from scripts.instruct_common import load_model, restore, nll
    from scripts.instruct_data import rows as data_rows
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from lm_eval.tasks.ifeval import utils as ifeval_utils, instructions, instructions_registry, instructions_util
    from langdetect import DetectorFactory
    DetectorFactory.seed = c.SEED
    ensure_ifeval_resources()
    started = time.monotonic()
    root = e.output_dir(exp) / ('eval-smoke' if limit else 'eval')
    root.mkdir(parents=True, exist_ok=True)
    gate = e.SANITY_ROOT / 'generation-protocol.json'
    if not gate.exists():
        raise ValueError('Run heldout generation audit before evaluation')
    gate_data = json.loads(gate.read_text())
    signature = e.design_signature()
    if gate_data['design_signature'] != signature or gate_data['generation'] != c.GENERATION:
        raise ValueError('Stale generation protocol audit')
    manifest = dict(design_signature=signature, experiment=exp.__dict__, limit=limit,
                    generation=c.GENERATION, datasets={c.MMLU_DATASET:c.MMLU_REVISION,c.IFEVAL_DATASET:c.IFEVAL_REVISION},
                    ifeval_code={Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                                 for m in (ifeval_utils,instructions,instructions_registry,instructions_util)},
                    lm_eval_version=importlib.metadata.version('lm_eval'))
    meta = root / 'manifest.json'
    if meta.exists() and json.loads(meta.read_text()) != manifest:
        raise ValueError('Instruct evaluation resume provenance mismatch')
    write_json(meta, manifest)
    model = load_model(exp.recipe, 'cuda').eval().requires_grad_(False)
    if exp.trainable:
        restore(e.output_dir(exp) / 'final.pt', model, 'cuda')
    tokenizer = AutoTokenizer.from_pretrained(c.MODEL, revision=c.REVISION)
    result = dict(design_signature=signature, experiment=exp.name)
    heldout = data_rows('test')
    result['assistant_nll'] = nll(model, heldout[:limit] if limit else heldout, torch.device('cuda'))
    write_json(root/'heldout.json', result['assistant_nll'])
    likelihood(model, tokenizer, root, limit)
    datasets = {'mmlu': list(load_dataset(c.MMLU_DATASET, 'all', revision=c.MMLU_REVISION, split='test')),
                'ifeval': list(load_dataset(c.IFEVAL_DATASET, revision=c.IFEVAL_REVISION, split='train'))}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for task, task_rows in datasets.items():
            task_rows = task_rows[:limit] if limit else task_rows
            for mode, settings in c.GENERATION.items():
                result[f'{task}-{mode}'] = evaluate_generation(model,tokenizer,task_rows,task,mode,settings,
                                    root,signature,c.SEED,started,c.EVAL_MAX_HOURS)
    write_json(root/'results.json', result)
    if not limit:
        write_json(e.output_dir(exp)/'evaluated.json', result)
    print(json.dumps(result), flush=True)


def audit():
    import experiments as e
    from scripts import instruct_config as c
    from scripts.instruct_common import load_model
    from scripts.instruct_data import rows
    from scripts.mmlu_generation import Decoder
    from transformers import AutoTokenizer
    ensure_ifeval_resources()
    model = load_model('original_ar', 'cuda').eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(c.MODEL, revision=c.REVISION)
    validation = rows('val')[:c.GENERATION_AUDIT_COUNT]
    if not validation:
        raise ValueError('No validation dialogs for generation audit')
    results = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for index, row in enumerate(validation):
            messages = row['messages'][:-1]
            for mode, settings in c.GENERATION.items():
                generated = generate(model,tokenizer,messages,mode=='thinking',settings,
                                      document_seed(c.SEED,'validation',mode,index))
                results.append(dict(index=index,mode=mode,**generated))
    # Compare cached inference to full forward on actual chat tokens, both model paths.
    ids = tokenizer.apply_chat_template(validation[0]['messages'][:-1], tokenize=True,
                        add_generation_prompt=True, enable_thinking=False, return_tensors='pt').to('cuda')[:, :c.AUDIT_LENGTH]
    parity = {}
    for sps in (False, True):
        model.sps = sps
        decoder = Decoder(model)
        actual = decoder.step(ids)
        expected = model(ids).logits[:, -1]
        torch.testing.assert_close(actual, expected, atol=c.GENERATION_PARITY_ATOL, rtol=c.GENERATION_PARITY_RTOL)
        parity[str(sps)] = (actual-expected).abs().max().item()
        # Exercise the continuation path too; prefill-only parity misses cache errors.
        for token in ids[0, -2:]:
            next_id = token.reshape(1, 1)
            actual = decoder.step(next_id)
            ids_extended = torch.cat((ids, next_id), dim=1)
            expected = model(ids_extended).logits[:, -1]
            torch.testing.assert_close(actual, expected, atol=c.GENERATION_PARITY_ATOL, rtol=c.GENERATION_PARITY_RTOL)
            ids = ids_extended
    write_json(e.SANITY_ROOT/'generation-audit-samples.json', results)
    write_json(e.SANITY_ROOT/'generation-protocol.json',dict(design_signature=e.design_signature(),
               generation=c.GENERATION, validation_count=len(validation), parity=parity,
               length_rates={mode:sum(x['stop']=='length' for x in results if x['mode']==mode)/len(validation)
                             for mode in c.GENERATION}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.audit:
        return audit()
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    import experiments as e
    matches = [x for x in e.experiment_configs if x.name == args.filter]
    if len(matches) != 1:
        parser.error('Select one exact experiment name')
    evaluate(matches[0], args.limit)

if __name__ == '__main__':
    main()
