"""Native-tokenizer, document-contained sequences with content-hash splits."""
import hashlib
import json
from pathlib import Path

import numpy as np

import experiments as e
from scripts import ar_sps_config as c


def split_for(text):
    digest = hashlib.sha256((c.DATA_SPLIT_SALT + text).encode()).hexdigest()
    value = int(digest[:16], 16) / 2**64
    return ('train' if value < c.TRAIN_FRACTION else 'val' if value < c.VAL_BOUNDARY else 'test'), digest


def prepare():
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    root = e.RUN_ROOT / 'data'
    root.mkdir(parents=True, exist_ok=True)
    signature = e.design_signature()
    marker = root / 'manifest.json'
    if marker.exists():
        result = json.loads(marker.read_text())
        if result['design_signature'] != signature:
            raise RuntimeError('Stale data manifest')
        for split, rows in result['rows'].items():
            if (root / f'{split}.bin').stat().st_size != rows*(c.CONTEXT+1)*4:
                raise RuntimeError('Incomplete data')
        return result
    tokenizer = AutoTokenizer.from_pretrained(c.MODEL, revision=c.REVISION)
    limits = dict(train=c.MAX_DATA_ROWS, val=c.VAL_ROWS, test=c.TEST_ROWS)
    counts = dict.fromkeys(limits, 0)
    handles = {s: (root/f'{s}.tmp').open('wb') for s in limits}
    seen = set()
    sources = []
    try:
        for filename in e.DATA_FILES:
            source = None
            for cache in e.EXISTING_HF_CACHES:
                candidate = try_to_load_from_cache(e.DATASET_REPO, filename, revision=e.DATASET_REVISION,
                                                   repo_type='dataset', cache_dir=str(cache))
                if isinstance(candidate, str) and Path(candidate).is_file():
                    source = candidate
                    break
            if source is None:
                source = hf_hub_download(e.DATASET_REPO, filename, revision=e.DATASET_REVISION,
                                         repo_type='dataset', cache_dir=str(e.HF_HOME/'hub'))
            sources.append(filename)
            for batch in pq.ParquetFile(source).iter_batches(batch_size=128, columns=['text']):
                texts = batch.column(0).to_pylist()
                for text in texts:
                    split, digest = split_for(text)
                    if digest in seen or counts[split] >= limits[split]:
                        continue
                    seen.add(digest)
                    ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
                    # No packed-document attention or EOS->next-document targets.
                    for start in range(0, len(ids)-c.CONTEXT, c.CONTEXT):
                        if counts[split] >= limits[split]:
                            break
                        row = np.asarray(ids[start:start+c.CONTEXT+1], dtype=np.uint32)
                        handles[split].write(row.tobytes())
                        counts[split] += 1
                if all(counts[s] >= limits[s] for s in limits):
                    break
            if all(counts[s] >= limits[s] for s in limits):
                break
    finally:
        for handle in handles.values():
            handle.close()
    if any(counts[s] < limits[s] for s in limits):
        raise RuntimeError(f'Insufficient document-contained rows: {counts}')
    for split in limits:
        (root/f'{split}.tmp').replace(root/f'{split}.bin')
    result = dict(design_signature=signature, model=c.MODEL, revision=c.REVISION,
                  dataset=e.DATASET_REPO, dataset_revision=e.DATASET_REVISION, files=sources,
                  rows=counts, row_length=c.CONTEXT+1, dtype='uint32',
                  sampling='full document-contained chunks; short documents and tails discarded',
                  contamination='No claim about source pretraining overlap; evaluation never used for selection')
    from scripts.mlspace_launch import store_json
    store_json(marker, result)
    return result


def rows(split):
    root = e.RUN_ROOT/'data'
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['design_signature'] != e.design_signature():
        raise RuntimeError('Stale data')
    return np.memmap(root/f'{split}.bin', mode='r', dtype=np.uint32).reshape(-1, c.CONTEXT+1)
