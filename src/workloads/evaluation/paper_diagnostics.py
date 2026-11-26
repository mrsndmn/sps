"""Read-only paired analysis of the paper's saved instruction evaluations.

No checkpoint selection, new sampling, or claims of population representativeness.
Intervals resample paired questions, not training seeds; categories are descriptive.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import instruct_config as c

ROOT = Path('/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/sps-artifacts')
BOOTSTRAPS = 20000
SEED = 9147
SUITES = {'constant512': 'ar-sps-qwen3-1.7b-instruct',
          'step384': 'qwen3-instruct-step384', 'cosine512': 'qwen3-instruct-cosine'}
ARMS = ('original_ar', 'staged_sps_ce', 'staged_sps_kd', 'continued_ar_ce', 'continued_ar_kd')
PAIRS = [(a, 'original_ar') for a in ARMS[1:]] + [
    ('staged_sps_ce', 'continued_ar_ce'), ('staged_sps_kd', 'continued_ar_kd')]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_rows(path, task):
    raw = path.read_bytes()
    result = {}
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b'\n'):
            raise ValueError(f'Incomplete record in {path}')
        row = json.loads(line)
        i = row['index']
        if type(i) is not int or i < 0 or i in result:
            raise ValueError(f'Invalid/duplicate index in {path}: {i}')
        if task == 'mmlu':
            if row['answer'] not in (None, 'A', 'B', 'C', 'D') or row['target'] not in 'ABCD':
                raise ValueError('Invalid MMLU label')
            if type(row['correct']) is not bool or row['correct'] != (row['answer'] == row['target']):
                raise ValueError('Inconsistent MMLU score')
        else:
            m = row['metrics']
            for level in ('strict', 'loose'):
                values = m[f'inst_level_{level}_acc']
                if not values or any(type(x) is not bool for x in values):
                    raise ValueError('Invalid instruction scores')
                if type(m[f'prompt_level_{level}_acc']) is not bool or m[f'prompt_level_{level}_acc'] != all(values):
                    raise ValueError('Inconsistent prompt score')
        result[i] = {k: v for k, v in row.items() if k not in ('tokens', 'raw_text', 'final_text', 'reasoning')}
    return result, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), count=len(result))


def align(arms, task, require_full=False):
    if not arms:
        raise ValueError('No arms')
    common = sorted(set.intersection(*(set(x) for x in arms.values())))
    if not common:
        raise ValueError('No shared questions')
    if require_full and any(set(x) != set(range(541)) for x in arms.values()):
        raise ValueError('IFEval requires all 541 questions in every arm')
    keys = ('doc_hash', 'seed', 'target', 'subject') if task == 'mmlu' else ('doc_hash', 'seed')
    # Check every overlap, not just the final common subset.
    identities = {}
    for rows in arms.values():
        for i, row in rows.items():
            identity = tuple(row[k] for k in keys)
            if i in identities and identities[i] != identity:
                raise ValueError(f'Question identity mismatch at {i}')
            identities[i] = identity
    return common


def paired(reference, candidate):
    a, b = np.asarray(reference, dtype=int), np.asarray(candidate, dtype=int)
    if a.shape != b.shape or a.ndim != 1 or not len(a) or not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError('Expected nonempty paired binary outcomes')
    counts = np.bincount(b-a+1, minlength=3)
    samples = np.random.default_rng(SEED).multinomial(len(a), counts / len(a), BOOTSTRAPS)
    delta = (samples[:, 2] - samples[:, 0]) * 100 / len(a)
    return dict(n=len(a), reference_pct=float(a.mean()*100), candidate_pct=float(b.mean()*100),
                delta_pp=float((b-a).mean()*100), ci95_pp=np.quantile(delta, [.025, .975]).tolist(),
                gained=int(counts[2]), lost=int(counts[0]))


def state(row):
    return 'invalid' if row['answer'] is None else ('correct' if row['correct'] else 'wrong')


def transitions(reference, candidate, ids):
    return {f'{a}->{b}': sum(state(reference[i]) == a and state(candidate[i]) == b for i in ids)
            for a in ('invalid', 'wrong', 'correct') for b in ('invalid', 'wrong', 'correct')}


def protocol(meta, task, mode):
    if meta['limit'] is not None or meta['generation'][mode] != c.family_config.GENERATION[mode]:
        raise ValueError('Evaluation protocol differs from the paper')
    if task == 'mmlu':
        if meta['datasets']['cais/mmlu'] != c.family_config.MMLU_REVISION:
            raise ValueError('MMLU revision mismatch')
        return {'generation': meta['generation'][mode], 'revision': meta['datasets']['cais/mmlu']}
    if meta['dataset'] != c.family_config.IFEVAL_DATASET or meta['revision'] != c.family_config.IFEVAL_REVISION or mode not in meta['modes']:
        raise ValueError('IFEval protocol mismatch')
    return {k: meta[k] for k in ('generation', 'revision')}


def run(root=ROOT, categories=None):
    report = {'scope': 'Exploratory paired question bootstrap; per-comparison 95% percentile intervals, no multiplicity correction; not training-seed uncertainty or equivalence evidence.',
              'bootstrap_replicates': BOOTSTRAPS, 'seed': SEED, 'ifeval': {}, 'mmlu': {}}
    for mode in c.family_config.GENERATION:
        arms, sources, protocols = {}, {}, []
        for arm in ARMS:
            path = root/'qwen3-instruct-ifeval/runs/out'/f'qwen3_instruct_{arm}--{mode}'/'eval'/f'ifeval-{mode}.jsonl'
            arms[arm], sources[arm] = read_rows(path, 'ifeval')
            meta = json.loads((path.parent/'manifest.json').read_text())
            sources[arm]['manifest'] = meta
            protocols.append(protocol(meta, 'ifeval', mode))
        if any(p != protocols[0] for p in protocols):
            raise ValueError('IFEval scoring provenance mismatch')
        ids = align(arms, 'ifeval', require_full=True)
        comparisons = []
        for candidate, reference in PAIRS:
            for metric in ('prompt_level_strict_acc', 'prompt_level_loose_acc'):
                comparisons.append(dict(candidate=candidate, reference=reference, metric=metric,
                    **paired([arms[reference][i]['metrics'][metric] for i in ids],
                             [arms[candidate][i]['metrics'][metric] for i in ids])))
        summary = {}
        for arm, rows in arms.items():
            summary[arm] = dict(length_count=sum(rows[i]['stop']=='length' for i in ids),
                               unclosed_thinking=sum(not rows[i]['thinking_closed'] for i in ids))
        category_results = []
        if categories is not None:
            for i in ids:
                if digest(categories[i]) != arms['original_ar'][i]['doc_hash']:
                    raise ValueError('IFEval category document hash mismatch')
                if any(len(rows[i]['metrics']['inst_level_strict_acc']) != len(categories[i]['instruction_id_list']) for rows in arms.values()):
                    raise ValueError('Instruction count mismatch')
            names = sorted({x for i in ids for x in categories[i]['instruction_id_list']})
            for name in names:
                positions = [(i, j) for i in ids for j, x in enumerate(categories[i]['instruction_id_list']) if x == name]
                entry = dict(category=name, instructions=len(positions), prompts=len({i for i,j in positions}))
                for arm, rows in arms.items():
                    entry[arm] = 100*sum(rows[i]['metrics']['inst_level_strict_acc'][j] for i,j in positions)/len(positions)
                category_results.append(entry)
        report['ifeval'][mode] = dict(sources=sources, comparisons=comparisons, diagnostics=summary, categories=category_results)

        arms, sources, protocols = {}, {}, []
        missing = []
        for schedule, suite in SUITES.items():
            expected = ARMS if schedule == 'constant512' else ARMS[1:]
            for arm in expected:
                pattern = f'qwen3_instruct_{arm}*/eval/mmlu-{mode}.jsonl'
                paths = list((root/suite/'runs/out').glob(pattern))
                label = 'original_ar' if arm == 'original_ar' else f'{schedule}/{arm}'
                if not paths:
                    missing.append(label)
                    continue
                if len(paths) != 1:
                    raise ValueError(f'Ambiguous MMLU input: {label}')
                rows, evidence = read_rows(paths[0], 'mmlu')
                if not rows:
                    missing.append(label)
                    continue
                arms[label], sources[label] = rows, evidence
                meta = json.loads((paths[0].parent/'manifest.json').read_text())
                sources[label]['manifest'] = meta
                protocols.append(protocol(meta, 'mmlu', mode))
        if any(p != protocols[0] for p in protocols):
            raise ValueError('MMLU protocol mismatch')
        if 'original_ar' not in arms:
            report['mmlu'][mode] = dict(status='No source answers; retention cannot be assessed', missing=missing, sources=sources)
            continue
        ids = align(arms, 'mmlu')
        comparisons = []
        for candidate, reference in PAIRS:
            a = f'constant512/{candidate}'
            b = 'original_ar' if reference == 'original_ar' else f'constant512/{reference}'
            if a not in arms or b not in arms:
                continue
            comparisons.append(dict(candidate=a, reference=b,
                **paired([arms[b][i]['correct'] for i in ids], [arms[a][i]['correct'] for i in ids]),
                transitions=transitions(arms[b], arms[a], ids)))
        report['mmlu'][mode] = dict(scope='Intersection of available arms; ordered subject prefix, not representative MMLU',
            missing=missing, indices=ids, sources=sources, comparisons=comparisons,
            subjects=dict(Counter(arms['original_ar'][i]['subject'] for i in ids)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/paper-diagnostics')
    parser.add_argument('--categories', action='store_true', help='Load pinned IFEval documents for category scores')
    args = parser.parse_args()
    categories = None
    if args.categories:
        from datasets import load_dataset
        categories = list(load_dataset(c.family_config.IFEVAL_DATASET, revision=c.family_config.IFEVAL_REVISION, split='train'))
    report = run(args.root, categories)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    entries = []
    for task in ('ifeval', 'mmlu'):
        for mode, group in report[task].items():
            for row in group.get('comparisons', []):
                entries.append(dict(task=task, mode=mode, metric=row.get('metric', 'accuracy'),
                    **{k:row[k] for k in ('candidate','reference','n','reference_pct','candidate_pct','delta_pp','gained','lost')},
                    ci_low=row['ci95_pp'][0], ci_high=row['ci95_pp'][1]))
    with (args.output/'paired.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(entries[0])); writer.writeheader(); writer.writerows(entries)
    for mode, group in report['ifeval'].items():
        if group['categories']:
            with (args.output/f'categories-{mode}.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(group['categories'][0])); writer.writeheader(); writer.writerows(group['categories'])
    print(json.dumps({'output':str(args.output), 'comparisons':entries}, indent=2))
