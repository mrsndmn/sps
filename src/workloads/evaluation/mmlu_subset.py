"""STEM selection and validated reuse from the original full-MMLU run."""
from experiments.evaluation_outputs import families
from workloads.evaluation.mmlu_results import matches_manifest
import hashlib
import json
from pathlib import Path
from workloads.evaluation.report_mmlu_stem import STEM_SUBJECTS, read_samples, summarize


def selected_indices(rows, stem=False, limit=None):
    indices = [i for i, row in enumerate(rows) if not stem or row['subject'] in STEM_SUBJECTS]
    return indices[:limit] if limit else indices


def reuse_samples(folder, exp, rows, indices):
    meta = folder / 'manifest.json'
    if not meta.exists():
        return []
    manifest = json.loads(meta.read_text())
    if not matches_manifest(manifest, exp, None):
        raise ValueError('Unexpected full-MMLU source manifest')
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
    for suite in ("mmlu-generation", "mmlu-stem-generation"):
        folder = families[suite].for_run(common_root, exp)
        for item in reuse_samples(folder, exp, rows, indices):
            index = item['index']
            if index in merged and merged[index] != item:
                raise ValueError('Conflicting answers across source evaluations')
            merged[index] = item
    result = list(merged.values())
    validate_saved(result, rows, indices)
    return result
