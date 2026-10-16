"""Reuse pinned HF shards, then stream GPT-2 tokens into atomic pilot memmaps."""
import json
import os
from pathlib import Path
import shutil

from experiments import (DATASET_REPO, DATASET_REVISION, DATA_FILES,
                        EXISTING_HF_CACHES, HF_HOME, RUN_ROOT, VAL_FRACTION, VAL_SEED,
                        TRAIN_TOKENS, MIN_FREE_BYTES)


def prepare():
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    import numpy as np
    import pyarrow.parquet as pq
    import tiktoken
    from src.data.prepare import _is_val

    dest = RUN_ROOT / 'data' / 'fineweb-edu'
    marker = dest / 'manifest.json'
    if marker.exists():
        manifest = json.loads(marker.read_text())
        if (manifest['revision'] != DATASET_REVISION or manifest['files'] != list(DATA_FILES)
                or manifest['tokenizer'] != 'gpt2' or manifest['repo'] != DATASET_REPO
                or manifest['val_seed'] != VAL_SEED or manifest['val_fraction'] != VAL_FRACTION):
            raise RuntimeError('Existing data manifest differs from the pilot configuration')
        for split in ('train', 'val'):
            if (dest / f'{split}.bin').stat().st_size != manifest['tokens'][split] * 2:
                raise RuntimeError(f'Incomplete {split} memmap')
        return manifest

    dest.mkdir(parents=True, exist_ok=True)
    # Deliberately conservative: source shards, uint16 output, temporary files,
    # checkpoints and downstream caches. No full 350BT download is attempted.
    required = MIN_FREE_BYTES
    free = shutil.disk_usage(dest).free
    if free < required:
        raise RuntimeError(f'Need {required / 1e9:.0f} GB free; have {free / 1e9:.1f} GB')
    enc = tiktoken.get_encoding('gpt2')
    counts = {'train': 0, 'val': 0}
    sources = []
    handles = {s: (dest / f'{s}.bin.tmp').open('wb') for s in counts}
    try:
        for filename in DATA_FILES:
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
    if counts['train'] < TRAIN_TOKENS or counts['val'] < 1048576:
        raise RuntimeError(f'Insufficient unique pilot data: {counts}')
    for split in counts:
        os.replace(dest / f'{split}.bin.tmp', dest / f'{split}.bin')
    manifest = dict(repo=DATASET_REPO, revision=DATASET_REVISION, files=list(DATA_FILES),
                    tokenizer='gpt2', dtype='uint16', tokens=counts, sources=sources,
                    val_fraction=VAL_FRACTION, val_seed=VAL_SEED,
                    ordering='source-file then source-row; training sampler randomizes chunks')
    tmp = marker.with_suffix('.tmp')
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(marker)
    return manifest


if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
