"""Measure the value of extending archived, length-capped Qwen teacher responses."""
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from scripts import instruct_config as c
from scripts import instruct_data as data
from scripts import vllm_sps_common as common
from scripts import vllm_sps_config as source
from scripts import mlspace_launch as launch


@dataclass(frozen=True)
class Probe:
    caps: tuple = (1024, 1536, 2048, 3072, 4096, 6144, 8192)
    per_source: int = 12
    retention: float = .95


experiment_configs = [Probe()]
ROOT = source.ROOT.parent / 'qwen3-teacher-cap-probe'
ARCHIVE = source.SOURCE_ROOT / 'runs/data'


def sample(config):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    train_ids = {r['source_id'] for r in common.data('train')}
    selection = json.loads((ARCHIVE/'selection.json').read_bytes())
    assert selection['design_signature'] == source.SOURCE_SIGNATURE
    candidates = {}
    for row in selection['records']:
        if row['split'] != 'train':
            continue
        key = row['source_id']
        if key in train_ids:
            train_ids.remove(key)
        elif row['mode'] == 'thinking':
            record = json.loads((ARCHIVE/'samples'/f'{key}.json').read_bytes())
            if (record['source_id'] != key or record['design_signature'] != source.SOURCE_SIGNATURE):
                raise ValueError('Archived response identity mismatch')
            if record.get('rejected') and len(record['generated_ids']) == config.caps[0]:
                candidates[key] = (row, record)
        if not train_ids:
            break
    groups = defaultdict(list)
    remaining = set(candidates)
    for name in c.DATA_FILES:
        path = hf_hub_download(c.DATASET_REPO, name, revision=c.DATASET_REVISION, repo_type='dataset')
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
            for doc in batch.to_pylist():
                if not data.valid_messages(doc['messages']):
                    continue
                key = data.group_id(doc['messages'])
                if key in remaining and doc['messages'] == candidates[key][0]['messages']:
                    groups[doc['source']].append(key)
                    remaining.remove(key)
            if not remaining:
                break
        if not remaining:
            break
    if remaining or train_ids or not groups:
        raise RuntimeError('Failed to reconstruct rejected training population')
    selected = []
    for domain, keys in sorted(groups.items()):
        ordered = sorted(keys, key=lambda k: hashlib.sha256(f'{c.SEED}:{k}'.encode()).digest())
        for key in ordered[:config.per_source]:
            selected.append((domain, len(keys), *candidates[key]))
    return selected


def run():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.device_count() != 1:
        raise RuntimeError('Requires one GPU')
    config, = experiment_configs
    selected = sample(config)
    tokenizer = AutoTokenizer.from_pretrained(c.MODEL, revision=c.REVISION)
    model = AutoModelForCausalLM.from_pretrained(c.MODEL, revision=c.REVISION,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    folder = ROOT / common.file_hash(Path(__file__))
    results = []
    for domain, population, row, archive in selected:
        key = row['source_id']
        path = folder/'samples'/f'{key}.json'
        prefix = archive['generated_ids']
        ident = dict(source_id=key, domain=domain, population=population,
                     input_hash=data.digest([row['messages'][:-1], prefix]), revision=c.REVISION)
        if path.exists():
            result = json.loads(path.read_bytes())
            if result['identity'] != ident:
                raise ValueError('Cached inputs differ')
        else:
            prompt = tokenizer.apply_chat_template(row['messages'][:-1], tokenize=True,
                add_generation_prompt=True, enable_thinking=True)
            ids = torch.tensor([prompt + prefix], device='cuda')
            torch.manual_seed(archive['seed'] ^ 0xCEA1)
            with torch.inference_mode():
                out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True,
                    temperature=c.TEACHER_TEMPERATURE, top_k=c.TEACHER_TOP_K, top_p=c.TEACHER_TOP_P,
                    max_new_tokens=config.caps[-1]-len(prefix), pad_token_id=tokenizer.eos_token_id)
            tokens = out[0, len(prompt):].tolist()
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            assistant = data.parse_thinking(text) if tokens[-1] in eos else None
            fits = False
            if assistant:
                try:
                    data.encode_example(tokenizer, row['messages'][:-1]+[assistant], 'thinking', key, c.CONTEXT)
                    fits = True
                except ValueError as err:
                    if 'exceeds context' not in str(err):
                        raise
            result = dict(identity=ident, tokens=len(tokens), complete=assistant is not None,
                          fits=fits, generated_ids=tokens, text=text)
            data.atomic_json(path, result)
        results.append(result)
    counts = defaultdict(int)
    for r in results:
        counts[r['identity']['domain']] += 1
    weights = [r['identity']['population']/counts[r['identity']['domain']] for r in results]
    curve = []
    for cap in config.caps:
        def average(values):
            return sum(w*v for w,v in zip(weights, values, strict=True))/sum(weights)
        curve.append(dict(cap=cap, complete=average(r['complete'] and r['tokens'] <= cap for r in results),
            trainable=average(r['fits'] and r['tokens'] <= cap for r in results),
            extra_tokens=average(min(cap, r['tokens'])-config.caps[0] for r in results)))
    recommendations = {k: next((p['cap'] for p in curve if p[k] >= config.retention*curve[-1][k]), None)
                       if curve[-1][k] else None for k in ('complete','trainable')}
    launch.store_json(folder/'report.json', dict(n=len(results), by_source=dict(counts),
        curve=curve, provisional_caps=recommendations))


if __name__ == '__main__':
    run()
