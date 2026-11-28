"""Reuse pinned HF shards, then stream GPT-2 tokens into atomic pilot memmaps."""
import json
import os
from pathlib import Path
import shutil

from orchestration.experiment_suite import (DATASET_REPO, DATASET_REVISION, DATA_FILES,
                        EXISTING_HF_CACHES, HF_HOME, RUN_ROOT, VAL_FRACTION, VAL_SEED,
                        collect_experiments, MIN_FREE_BYTES)


def prepare():
    dest = RUN_ROOT / 'data' / 'fineweb-edu'
    if (dest / 'manifest.json').is_file():
        return _prepare(dest)
    from orchestration.run_directory import preparation_lock
    with preparation_lock(dest):
        return _prepare(dest)


def checked_manifest(dest, files):
    manifest = json.loads((dest / 'manifest.json').read_text())
    expected = dict(revision=DATASET_REVISION, files=list(files), tokenizer='gpt2',
                    dtype='uint16', repo=DATASET_REPO, val_seed=VAL_SEED, val_fraction=VAL_FRACTION)
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f'Unexpected corpus manifest at {dest}')
    for split in ('train', 'val'):
        if (dest / f'{split}.bin').stat().st_size != manifest['tokens'][split] * 2:
            raise RuntimeError(f'Incomplete {split} memmap at {dest}')
    return manifest


def _prepare(dest):
    return prepare_memmaps(dest, DATA_FILES, MIN_FREE_BYTES,
                           max(run.training.max_tokens for run in collect_experiments()))


def prepare_memmaps(dest, files, required, train_tokens, *, prefix_dir=None, prefix_files=(), parts=()):

    marker = dest / 'manifest.json'
    if marker.exists():
        return checked_manifest(dest, files)
    if tuple(files[:len(prefix_files)]) != tuple(prefix_files) or len(set(files)) != len(files):
        raise ValueError('Expected distinct shards starting with the cached prefix')
    covered = len(prefix_files) if prefix_dir is not None else 0
    for _, part_files in parts:
        if tuple(files[covered:covered + len(part_files)]) != tuple(part_files):
            raise ValueError("Tokenized parts do not match the ordered shard list")
        covered += len(part_files)
    dest.mkdir(parents=True, exist_ok=True)
    # Deliberately conservative: source shards, uint16 output, temporary files,
    # checkpoints and downstream caches. No full 350BT download is attempted.
    free = shutil.disk_usage(dest).free
    if free < required:
        raise RuntimeError(f'Need {required / 1e9:.0f} GB free; have {free / 1e9:.1f} GB')
    if covered < len(files):
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
        import numpy as np
        import pyarrow.parquet as pq
        import tiktoken
        from data.prepare import _is_val

        enc = tiktoken.get_encoding('gpt2')
    counts = {'train': 0, 'val': 0}
    sources = []
    handles = {s: (dest / f'{s}.bin.tmp').open('wb') for s in counts}
    try:
        segments = ([(prefix_dir, prefix_files)] if prefix_dir is not None else []) + list(parts)
        for part_dir, part_files in segments:
            prefix = checked_manifest(part_dir, part_files)
            for split, handle in handles.items():
                with (part_dir / f'{split}.bin').open('rb') as source_bin:
                    shutil.copyfileobj(source_bin, handle, length=8 << 20)
                counts[split] += prefix['tokens'][split]
            sources.extend(prefix['sources'])
        for filename in files[covered:]:
            source = None
            for cache in EXISTING_HF_CACHES:
                candidate = try_to_load_from_cache(DATASET_REPO, filename,
                    revision=DATASET_REVISION, repo_type='dataset', cache_dir=str(cache))
                if isinstance(candidate, str) and Path(candidate).is_file():
                    source = candidate
                    break
            if source is None:
                source = hf_hub_download(DATASET_REPO, filename, repo_type='dataset',
                    revision=DATASET_REVISION, cache_dir=str(HF_HOME / 'hub'))
            print(f'Data source: {source}', flush=True)
            parquet = pq.ParquetFile(source)
            if 'score' not in parquet.schema.names:
                raise RuntimeError('Expected FineWeb-Edu education-score column')
            sources.append({'file': filename, 'path': source, 'bytes': Path(source).stat().st_size})
            for batch in parquet.iter_batches(batch_size=512, columns=['text', 'id']):
                records = batch.to_pydict()
                tokenized = enc.encode_ordinary_batch(records['text'], num_threads=8)
                for doc_id, tokens in zip(records['id'], tokenized):
                    split = 'val' if _is_val(doc_id, VAL_SEED, VAL_FRACTION) else 'train'
                    tokens.append(enc.eot_token)
                    handles[split].write(np.asarray(tokens, dtype=np.uint16).tobytes())
                    counts[split] += len(tokens)
            print(f'Tokens so far: {counts}', flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    if counts['train'] < train_tokens or counts['val'] < 1048576:
        raise RuntimeError(f'Insufficient unique pilot data: {counts}')
    for split in counts:
        os.replace(dest / f'{split}.bin.tmp', dest / f'{split}.bin')
    manifest = dict(repo=DATASET_REPO, revision=DATASET_REVISION, files=list(files),
                    tokenizer='gpt2', dtype='uint16', tokens=counts, sources=sources,
                    val_fraction=VAL_FRACTION, val_seed=VAL_SEED,
                    ordering='source-file then source-row; training sampler randomizes chunks')
    tmp = marker.with_suffix('.tmp')
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(marker)
    return manifest


def main():
    print(json.dumps(prepare(), indent=2))
