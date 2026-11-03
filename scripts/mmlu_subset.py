"""STEM selection and validated reuse from the original full-MMLU run."""
import hashlib
import json
from pathlib import Path
from scripts.report_mmlu_stem import STEM_SUBJECTS, SOURCE_SIGNATURE, STEM_SIGNATURE, read_samples, summarize

GENERATION_SHA256 = '11504a2e36a7bfb43d6cadccfa46d6d6f785cae4f9e9f5a200be277affae88a8'


def selected_indices(rows, stem=False, limit=None):
    indices = [i for i, row in enumerate(rows) if not stem or row['subject'] in STEM_SUBJECTS]
    return indices[:limit] if limit else indices


def reuse_samples(folder, exp, rows, indices, *, signature=SOURCE_SIGNATURE):
    meta = folder / 'manifest.json'
    if not meta.exists():
        return []
    manifest = json.loads(meta.read_text())
    if (manifest.get('design_signature') != signature or manifest.get('limit') is not None
            or manifest.get('experiment') != exp.__dict__):
        raise ValueError('Unexpected full-MMLU source manifest')
    from scripts import mmlu_config as c
    generation = Path(__file__).with_name('mmlu_generation.py')
    if (hashlib.sha256(generation.read_bytes()).hexdigest() != GENERATION_SHA256 or
            c.MAX_NEW_TOKENS != 512 or c.SEED != 1337 or c.DATASET != 'cais/mmlu' or
            c.REVISION != 'c30699e8356da336a370243923dbaf21066bb9fe'):
        raise ValueError('Generation protocol changed; cannot reuse old answers')
    samples = read_samples(folder / 'samples.jsonl')
    summarize(samples, rows)  # Validate document hashes, targets, subjects and unique IDs.
    selected = set(indices)
    return [item for item in samples if item['index'] in selected]


def validate_saved(samples, rows, indices):
    expected = set(indices)
    seen = set()
    for item in samples:
        index = item['index']
        if index not in expected or index in seen:
            raise ValueError('Unexpected or duplicate saved sample index')
        seen.add(index)
        row = rows[index]
        digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        if (item['doc_hash'] != digest or item['subject'] != row['subject'] or
                item['target'] != 'ABCD'[row['answer']] or
                item['correct'] != (item['answer'] == item['target'])):
            raise ValueError('Resume sample provenance mismatch')
    return seen




def reuse_full_and_stem(common_root, exp, rows, indices):
    merged = {}
    for suite, signature in [('mmlu-generation', SOURCE_SIGNATURE),
                             ('mmlu-stem-generation', STEM_SIGNATURE)]:
        folder = common_root / suite / 'runs/out' / exp.name / 'full'
        for item in reuse_samples(folder, exp, rows, indices, signature=signature):
            index = item['index']
            if index in merged and merged[index] != item:
                raise ValueError('Conflicting answers across source evaluations')
            merged[index] = item
    result = list(merged.values())
    validate_saved(result, rows, indices)
    return result
