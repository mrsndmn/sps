"""Paired question analysis of fixed provenance-control endpoints; no selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts import paper_diagnostics as d
from scripts import provenance_evaluate as evaluation

METRICS = ('prompt_level_strict_acc', 'prompt_level_loose_acc')
ARMS = tuple(e.name for e in evaluation.train.experiment_configs)
PAIRS = [('continued_ar-teacher','continued_ar-corpus'), ('staged_sps-teacher','staged_sps-corpus'),
         ('staged_sps-corpus','continued_ar-corpus'), ('staged_sps-teacher','continued_ar-teacher')]
ROOT = evaluation.ROOT / 'reports'


def interaction(values):
    values = np.asarray(values, dtype=int)
    if values.ndim != 1 or not len(values) or not np.isin(values, [-2,-1,0,1,2]).all():
        raise ValueError('Invalid paired interaction outcomes')
    support, counts = np.unique(values, return_counts=True)
    draws = np.random.default_rng(d.SEED).multinomial(len(values), counts/len(values), d.BOOTSTRAPS)
    deltas = draws @ support * 100 / len(values)
    return dict(delta_pp=float(values.mean()*100), ci95_pp=np.quantile(deltas,[.025,.975]).tolist(), n=len(values))


def load_control(exp):
    root = evaluation.output_dir(exp)
    result = json.loads((root/'evaluated.json').read_text())
    ident = result['identity']
    manifest = json.loads((root/'eval/manifest.json').read_text())
    if ident != manifest or ident['experiment'] != exp.__dict__ or not result['passed']:
        raise ValueError('Completed control identity mismatch')
    rows, evidence = d.read_rows(root/'eval'/f'ifeval-{exp.mode}.jsonl','ifeval')
    if evidence['sha256'] != result['samples_sha256']:
        raise ValueError('Saved control answers changed')
    metrics = evaluation.evaluator.aggregate(list(rows.values()),'ifeval')
    if result['metrics'] != dict(design_signature=d.digest(ident), **metrics):
        raise ValueError('Saved aggregate differs from answer scores')
    return rows, result, evidence


def run(mode):
    arms, results, evidence = {}, {}, {}
    for exp in evaluation.experiment_configs:
        if exp.mode == mode:
            arms[exp.arm], results[exp.arm], evidence[exp.arm] = load_control(exp)
    protocols = [r['identity']['protocol'] for r in results.values()]
    if any(p != protocols[0] for p in protocols):
        raise ValueError('Control protocols differ')
    codes = [r['identity']['code'] for r in results.values()]
    if any(code != codes[0] for code in codes):
        raise ValueError('Control evaluation implementations differ')
    preparations = {r['identity']['preparation_sha256'] for r in results.values()}
    if len(preparations) != 1:
        raise ValueError('Control prepared data differs')
    baseline = d.ROOT/'qwen3-instruct-ifeval/runs/out'/f'qwen3_instruct_original_ar--{mode}'/'eval'
    meta = json.loads((baseline/'manifest.json').read_text())
    old = d.protocol(meta,'ifeval',mode)
    p = protocols[0]
    if (old['generation'] != p['generation'] or old['revision'] != p['revision']
            or old['evaluator_code'] != p['ifeval_code'] or p['dataset'] != d.c.IFEVAL_DATASET):
        raise ValueError('Source baseline protocol differs')
    arms['original_ar'], evidence['original_ar'] = d.read_rows(baseline/f'ifeval-{mode}.jsonl','ifeval')
    ids = d.align(arms,'ifeval',require_full=True)
    comparisons, interactions = [], []
    for metric in METRICS:
        scores = {name:np.array([rows[i]['metrics'][metric] for i in ids],dtype=int) for name,rows in arms.items()}
        for candidate,reference in PAIRS + [(a,'original_ar') for a in ARMS]:
            comparisons.append(dict(metric=metric,candidate=candidate,reference=reference,
                                    **d.paired(scores[reference],scores[candidate])))
        interactions.append(dict(metric=metric, definition='SPS teacher-minus-corpus minus AR teacher-minus-corpus',
            **interaction(scores['staged_sps-teacher']-scores['staged_sps-corpus']
                          -scores['continued_ar-teacher']+scores['continued_ar-corpus'])))
    summary={name:evaluation.evaluator.aggregate(list(rows.values()),'ifeval') for name,rows in arms.items()}
    report=dict(mode=mode,summary=summary,comparisons=comparisons,interactions=interactions,
        crossed_heldout={a:r['crossed_heldout'] for a,r in results.items()},evidence=evidence,
        identities={a:r['identity'] for a,r in results.items()},baseline_manifest=meta,
        bootstrap_replicates=d.BOOTSTRAPS,seed=d.SEED,
        scope='Exploratory paired question bootstrap with per-comparison 95% intervals, no multiplicity correction; one training/evaluation seed. Not equivalence evidence or length/token-exposure-controlled provenance attribution. Crossed NLL compares models only within a fixed target corpus.')
    evaluation.launch.store_json(ROOT/f'{mode}.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=evaluation.ic.MODES,default='nonthinking')
    args=parser.parse_args()
    r=run(args.mode)
    print(json.dumps(dict(summary=r['summary'],comparisons=r['comparisons'],interactions=r['interactions']),indent=2))


if __name__=='__main__':main()
