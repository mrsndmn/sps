"""Read-only STEM aggregation of the pinned MMLU reasoning evaluation."""
import hashlib
import json
from pathlib import Path

# Official categories: https://github.com/hendrycks/test/blob/master/categories.py
STEM_SUBJECTS = frozenset('''abstract_algebra astronomy college_biology college_chemistry
college_computer_science college_mathematics college_physics computer_security
conceptual_physics electrical_engineering elementary_mathematics high_school_biology
high_school_chemistry high_school_computer_science high_school_mathematics
high_school_physics high_school_statistics machine_learning'''.split())
SOURCE_SIGNATURE = '78399971569afb2da34334515f2a4943e4429212c3a8077a82fd059e7045bf58'
STEM_SIGNATURE = '1e42f6ab6d346a07097daca5b12f7f484c41bb1775c15c7f1271b6af7b04ab4e'


def read_samples(path):
    """Ignore an unfinished final write without modifying a running worker's file."""
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    if lines and not lines[-1].endswith(b'\n'):
        lines.pop()
    return [json.loads(line) for line in lines]


def summarize(samples, dataset):
    expected = {i for i, row in enumerate(dataset) if row['subject'] in STEM_SUBJECTS}
    seen = set()
    selected = []
    for item in samples:
        index = item['index']
        if index in seen or not 0 <= index < len(dataset):
            raise ValueError('Duplicate or invalid sample index')
        seen.add(index)
        row = dataset[index]
        digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        if (item['doc_hash'] != digest or item['subject'] != row['subject'] or
                item['target'] != 'ABCD'[row['answer']]):
            raise ValueError('Sample provenance mismatch')
        if index in expected:
            selected.append(item)
    complete = bool(expected) and expected <= seen
    result = dict(expected=len(expected), completed=len(selected), complete=complete,
                  accuracy=None, macro_accuracy=None, invalid_rate=None, length_rate=None,
                  context_overflow_rate=None)
    if complete:
        correct = lambda x: x['answer'] == x['target']
        subjects = sorted({x['subject'] for x in selected})
        per_subject = {s: sum(correct(x) for x in selected if x['subject'] == s) /
                       sum(x['subject'] == s for x in selected) for s in subjects}
        result.update(accuracy=sum(map(correct, selected))/len(selected),
                      macro_accuracy=sum(per_subject.values())/len(per_subject),
                      per_subject=per_subject,
                      invalid_rate=sum(x['answer'] is None for x in selected)/len(selected),
                      length_rate=sum(x['stop'] == 'length' for x in selected)/len(selected),
                      context_overflow_rate=sum(x['stop'] == 'context_overflow' for x in selected)/len(selected))
    return result


def main():
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=['mmlu-generation', 'mmlu-stem-generation'], default='mmlu-generation')
    args = parser.parse_args()
    os.environ['SPS_EXPERIMENT_SUITE'] = args.suite
    from datasets import load_dataset
    from experiments import COMMON_ROOT
    from scripts.mmlu_config import DATASET, REVISION, experiment_configs
    dataset = list(load_dataset(DATASET, 'all', revision=REVISION, split='test'))
    root = COMMON_ROOT / args.suite
    signature = SOURCE_SIGNATURE if args.suite == 'mmlu-generation' else STEM_SIGNATURE
    results = {}
    for exp in experiment_configs:
        folder = root / 'runs/out' / exp.name / 'full'
        meta = folder / 'manifest.json'
        if not meta.exists():
            samples = []
        else:
            manifest = json.loads(meta.read_text())
            if (manifest['design_signature'] != signature or manifest['limit'] is not None
                    or manifest['experiment'] != exp.__dict__):
                raise ValueError(f'Unexpected source evaluation: {exp.name}')
            samples = read_samples(folder / 'samples.jsonl')
        results[exp.name] = summarize(samples, dataset)
    report = dict(benchmark='mmlu_stem_generation', dataset=DATASET, revision=REVISION,
                  source_signature=signature, subjects=sorted(STEM_SUBJECTS), results=results)
    output = root / 'reports/mmlu_stem.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2)); temporary.replace(output)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
