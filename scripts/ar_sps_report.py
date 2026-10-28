"""Strict observed retention with paired, per-task bootstrap uncertainty."""
import json
import numpy as np
import experiments as e
from scripts import ar_sps_config as c


def paired_interval(left, right):
    if len(left) != len(right) or not len(left):
        raise ValueError('Paired measurements must be nonempty and equal length')
    delta = np.asarray(right, dtype=float)-np.asarray(left, dtype=float)
    rng = np.random.default_rng(c.SEED)
    means = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(c.BOOTSTRAP_REPS)]
    alpha = (1-c.CONFIDENCE)/2
    return [float(v) for v in np.quantile(means, [alpha, 1-alpha])]


def samples(result, task, metric):
    values = result['samples'][task]
    key = metric.split(',')[0]
    paired = {}
    for row in values:
        identity = str(row['doc_id'])
        if identity in paired:
            raise ValueError('Duplicate document result')
        paired[identity] = (row.get('doc_hash'), float(row[key]))
    return paired


def compare(baseline, candidate):
    if not (baseline/'complete.json').exists() or not (candidate/'complete.json').exists():
        raise RuntimeError('Evaluation incomplete; no retention verdict')
    for path in (baseline, candidate):
        if json.loads((path/'complete.json').read_text())['design_signature'] != e.design_signature():
            raise RuntimeError('Stale evaluation')
    row = dict(variant=str(candidate), tasks={})
    passed = True
    for task, metric in c.PRIMARY_METRICS.items():
        left = json.loads((baseline/f'{task}.json').read_text())
        right = json.loads((candidate/f'{task}.json').read_text())
        a, b = samples(left, task, metric), samples(right, task, metric)
        if a.keys() != b.keys() or any(a[k][0] != b[k][0] for k in a):
            raise RuntimeError('Evaluation documents do not match')
        ids = sorted(a)
        source_score = float(left['results'][task][metric])
        score = float(right['results'][task][metric])
        passed &= score >= source_score
        row['tasks'][task] = dict(source=source_score, score=score, delta=score-source_score,
                                  paired_ci=paired_interval([a[k][1] for k in ids], [b[k][1] for k in ids]))
    a = json.loads((baseline/'heldout.json').read_text())
    b = json.loads((candidate/'heldout.json').read_text())
    passed &= b['nll'] <= a['nll']+c.NLL_ATOL
    row.update(source_nll=a['nll'], nll=b['nll'], nll_ci=paired_interval(a['per_sequence'], b['per_sequence']),
               observed_retention=bool(passed), interpretation='Point-estimate criterion; one seed; CIs are per-task, not simultaneous proof')
    return row


def main():
    baseline = e.RUN_ROOT/'out/smollm2_original_ar/eval/source'
    reports = []
    for exp in e.collect_experiments():
        for marker in sorted((e.output_dir(exp)/'eval').glob('*/complete.json')):
            reports.append(compare(baseline, marker.parent))
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
