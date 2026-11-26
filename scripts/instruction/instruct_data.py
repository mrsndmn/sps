"""Pinned chat data with group splits, assistant masks and resumable teacher samples.

Only the final assistant turn is supervised. No packing, padding or truncation is
performed here; rejected examples and generation outcomes remain auditable.
"""
import hashlib
import json
import os
import re
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def group_id(messages):
    """Shared first-user prompts stay together even with different responses."""
    return digest(normalize(next(m['content'] for m in messages if m['role'] == 'user')))


def split_for(group, salt, train_fraction, val_boundary):
    x = int(hashlib.sha256((salt + group).encode()).hexdigest()[:16], 16) / 2**64
    return 'train' if x < train_fraction else 'val' if x < val_boundary else 'test'


def valid_messages(messages):
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    if any(not isinstance(m, dict) or m.get('role') not in ('system', 'user', 'assistant')
           or not isinstance(m.get('content'), str) or not m['content'].strip()
           or m.get('tool_calls') for m in messages):
        return False
    body = messages[1:] if messages[0]['role'] == 'system' else messages
    return (len(body) >= 2 and len(body) % 2 == 0
            and all(m['role'] == ('user' if i % 2 == 0 else 'assistant') for i, m in enumerate(body)))


class OverlapFilter:
    """Conservative normalized exact / long contiguous overlap with eval prompts."""
    def __init__(self, texts, ngram=13):
        self.ngram = ngram
        self.short = set()
        self.grams = set()
        for text in texts:
            words = normalize(text).split()
            if not words:
                continue
            if len(words) < ngram:
                self.short.add(' '.join(words))
            else:
                self.grams.update(tuple(words[i:i+ngram]) for i in range(len(words)-ngram+1))

    def matches(self, text):
        normalized = normalize(text)
        words = normalized.split()
        padded = ' ' + normalized + ' '
        return (any(' ' + s + ' ' in padded for s in self.short)
                or any(tuple(words[i:i+self.ngram]) in self.grams
                       for i in range(len(words)-self.ngram+1)))


def encode_example(tokenizer, messages, mode, source_id, context):
    if not valid_messages(messages):
        raise ValueError('Unsupported conversation structure')
    thinking = mode == 'thinking'
    if mode not in ('thinking', 'nonthinking'):
        raise ValueError('Unknown mode')
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=thinking)
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=False, enable_thinking=thinking)
    if not text.startswith(prefix):
        raise ValueError('Native template changed assistant prefix')
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded['input_ids']
    if len(ids) > context:
        raise ValueError('Conversation exceeds context; no truncation')
    mask = [int(start >= len(prefix) and end > start) for start, end in encoded['offset_mapping']]
    if not any(mask[1:]):
        raise ValueError('Empty assistant target')
    return dict(input_ids=ids, loss_mask=mask, mode=mode, source_id=source_id,
                group_id=group_id(messages), messages=messages, target_tokens=sum(mask))


def parse_thinking(text):
    """Require one closed nonempty reasoning block and a nonempty final answer."""
    match = re.fullmatch(r'\s*<think>\s*(.*?)\s*</think>\s*(.+?)\s*', text, flags=re.S)
    if not match or not match[1].strip() or '<think>' in match[1] or '</think>' in match[2] or '<think>' in match[2]:
        return None
    return {'role': 'assistant', 'reasoning_content': match[1].strip(), 'content': match[2].strip()}


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp.replace(path)


def write_jsonl(path, records):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    temp.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _config():
    import experiments as e
    from scripts import instruct_config as c
    return e, c


def _overlap_texts(c):
    """Explicit pinned evaluation datasets; no reliance on mutable local caches."""
    from datasets import load_dataset
    for spec in c.CONTAMINATION_DATASETS:
        dataset = load_dataset(spec['repo'], spec.get('config'), revision=spec['revision'], split=spec['split'],
                               trust_remote_code=spec.get('trust_remote_code', False))
        for row in dataset:
            for field in spec['fields']:
                value = row.get(field)
                if isinstance(value, str):
                    yield value
    for name in getattr(c, 'CONTAMINATION_FILES', ()):
        for line in Path(name).read_text().splitlines():
            row = json.loads(line)
            for field in ('question', 'prompt', 'text'):
                if isinstance(row.get(field), str):
                    yield row[field]


def _select(tokenizer, root, signature, c):
    selection = root/'selection.json'
    if selection.exists():
        obj = json.loads(selection.read_text())
        if obj['design_signature'] != signature:
            raise RuntimeError('Stale instruct selection')
        return obj['records']
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    overlap = OverlapFilter(_overlap_texts(c))
    counts = {s: 0 for s in c.DATA_ROWS}
    seen = set()
    selected = []
    rejected = {}
    for filename in c.DATA_FILES:
        path = hf_hub_download(c.DATASET_REPO, filename, revision=c.DATASET_REVISION, repo_type='dataset')
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            for example in batch.to_pylist():
                messages = example.get('messages')
                reason = None
                if not valid_messages(messages):
                    reason = 'unsupported_messages'
                elif any('<|' in m['content'] or '<think>' in m['content'] or '</think>' in m['content'] for m in messages):
                    reason = 'reserved_delimiters'
                if reason:
                    rejected[reason] = rejected.get(reason, 0)+1
                    continue
                group = group_id(messages)
                if group in seen:
                    continue
                seen.add(group)
                split = split_for(group, c.DATA_SPLIT_SALT, c.TRAIN_FRACTION, c.VAL_BOUNDARY)
                if counts[split] >= c.DATA_ROWS[split] * c.DATA_CANDIDATE_MULTIPLIER:
                    continue
                if overlap.matches('\n'.join(m['content'] for m in messages)):
                    rejected['benchmark_overlap'] = rejected.get('benchmark_overlap', 0)+1
                    continue
                mode = 'thinking' if int(group[:16], 16)/2**64 < c.THINKING_FRACTION else 'nonthinking'
                try:
                    row = encode_example(tokenizer, messages, 'nonthinking', group, c.CONTEXT)
                    if mode == 'thinking':
                        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=True)
                        if len(prompt) + c.TEACHER_MAX_NEW_TOKENS > c.CONTEXT:
                            raise ValueError('Thinking generation budget exceeds context')
                except ValueError:
                    rejected['length_or_template'] = rejected.get('length_or_template', 0)+1
                    continue
                selected.append(dict(source_id=group, group_id=group, split=split, mode=mode, messages=messages))
                counts[split] += 1
            if all(counts[s] >= c.DATA_ROWS[s] * c.DATA_CANDIDATE_MULTIPLIER for s in counts):
                break
        if all(counts[s] >= c.DATA_ROWS[s] * c.DATA_CANDIDATE_MULTIPLIER for s in counts):
            break
    if not all(counts[s] >= c.DATA_ROWS[s] for s in counts):
        raise RuntimeError(f'Insufficient selected conversations: {counts}')
    atomic_json(selection, dict(design_signature=signature, records=selected, selected=counts, rejected=rejected))
    return selected


def prepare():
    """Called by all torchrun ranks; independent model per GPU, no DDP teacher."""
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer, AutoModelForCausalLM
    e, c = _config()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    if world > 1 and not dist.is_initialized():
        from datetime import timedelta
        dist.init_process_group('gloo', timeout=timedelta(hours=24))
    def barrier():
        if world > 1:
            dist.barrier()
    root = e.RUN_ROOT/'data'
    root.mkdir(parents=True, exist_ok=True)
    signature = e.design_signature()
    if (root/'manifest.json').exists():
        for split in c.DATA_ROWS:
            list(rows(split))
        return json.loads((root/'manifest.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(c.MODEL, revision=c.REVISION)
    if rank == 0:
        _select(tokenizer, root, signature, c)
    barrier()
    selection = json.loads((root/'selection.json').read_text())
    if selection['design_signature'] != signature:
        raise RuntimeError('Selection signature mismatch')
    samples = root/'samples'
    samples.mkdir(exist_ok=True)
    device = f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}'
    torch.cuda.set_device(device)
    teacher = None
    for i, source in enumerate(selection['records']):
        if i % world != rank:
            continue
        output = samples/f'{source["source_id"]}.json'
        if output.exists():
            cached = json.loads(output.read_text())
            if cached['design_signature'] != signature:
                raise RuntimeError('Stale teacher sample')
            continue
        messages = source['messages']
        record = dict(design_signature=signature, source_id=source['source_id'], split=source['split'], mode=source['mode'])
        if source['mode'] == 'thinking':
            if teacher is None:
                teacher = AutoModelForCausalLM.from_pretrained(c.MODEL, revision=c.REVISION, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
                teacher.requires_grad_(False)
            seed = (c.SEED + int(source['source_id'][:8], 16)) % 2**31
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=True, return_tensors='pt').to(device)
            with torch.inference_mode():
                generated = teacher.generate(prompt, attention_mask=torch.ones_like(prompt), do_sample=True,
                    max_new_tokens=c.TEACHER_MAX_NEW_TOKENS, temperature=c.TEACHER_TEMPERATURE,
                    top_p=c.TEACHER_TOP_P, top_k=c.TEACHER_TOP_K, pad_token_id=tokenizer.eos_token_id)
            tokens = generated[0, prompt.shape[1]:].tolist()
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            assistant = parse_thinking(text)
            eos = teacher.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            record.update(generated_ids=tokens, generated_text=text, seed=seed)
            if not tokens or tokens[-1] not in eos or assistant is None:
                record['rejected'] = 'incomplete_or_malformed_thinking'
                atomic_json(output, record)
                continue
            messages = messages[:-1] + [assistant]
        try:
            row = encode_example(tokenizer, messages, source['mode'], source['source_id'], c.CONTEXT)
            record['row'] = row
        except ValueError as ex:
            record['rejected'] = str(ex)
        atomic_json(output, record)
    barrier()
    if rank == 0:
        accepted = {s: [] for s in c.DATA_ROWS}
        rejected = {}
        for source in selection['records']:
            record = json.loads((samples/f'{source["source_id"]}.json').read_text())
            if record['design_signature'] != signature:
                raise RuntimeError('Sample signature mismatch')
            split = source['split']
            if 'row' in record and len(accepted[split]) < c.DATA_ROWS[split]:
                accepted[split].append(record['row'])
            elif 'rejected' in record:
                reason = record['rejected']
                rejected[reason] = rejected.get(reason, 0)+1
        counts = {s: len(v) for s, v in accepted.items()}
        if any(counts[s] < c.DATA_ROWS[s]*c.MIN_DATA_FRACTION for s in counts):
            raise RuntimeError(f'Too few accepted rows: {counts}')
        mode_counts = {s: {m: sum(r['mode'] == m for r in v) for m in ('thinking','nonthinking')} for s,v in accepted.items()}
        if any(not count for modes in mode_counts.values() for count in modes.values()):
            raise RuntimeError(f'Missing required training/evaluation mode: {mode_counts}')
        import random
        for split, records in accepted.items():
            random.Random(c.SEED + int(digest(split)[:8], 16)).shuffle(records)
            write_jsonl(root/f'{split}.jsonl', records)
        manifest = dict(design_signature=signature, model=c.MODEL, revision=c.REVISION,
            dataset=c.DATASET_REPO, dataset_revision=c.DATASET_REVISION, dataset_files=c.DATA_FILES,
            tokenizer_template_hash=digest(tokenizer.chat_template), rows=counts, modes=mode_counts,
            rejected=rejected, selection_rejected=selection.get('rejected', {}),
            sha256={s:file_hash(root/f'{s}.jsonl') for s in counts},
            contamination_datasets=c.CONTAMINATION_DATASETS,
            scope='Exact normalized/13-token overlap filter only; pretraining contamination unknown')
        atomic_json(root/'manifest.json', manifest)
    barrier()
    return json.loads((root/'manifest.json').read_text())


def rows(split, verify=True):
    e, c = _config()
    if split not in c.DATA_ROWS:
        raise ValueError(f'Unknown split {split}')
    root = e.RUN_ROOT/'data'
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['design_signature'] != e.design_signature() or manifest['model'] != c.MODEL or manifest['revision'] != c.REVISION:
        raise RuntimeError('Stale instruct data provenance')
    path = root/f'{split}.jsonl'
    if verify and file_hash(path) != manifest['sha256'][split]:
        raise RuntimeError('Instruct data checksum mismatch')
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) != manifest['rows'][split]:
        raise RuntimeError('Instruct data row count mismatch')
    return records


if __name__ == '__main__':
    prepare()
