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


def compare(baseline, candidate, *, matched_exposure=False):
    if not (baseline/'complete.json').exists() or not (candidate/'complete.json').exists():
        raise RuntimeError('Evaluation incomplete; no retention verdict')
    for path in (baseline, candidate):
        if json.loads((path/'complete.json').read_text())['design_signature'] != e.design_signature():
            raise RuntimeError('Stale evaluation')
    row = dict(baseline=str(baseline), variant=str(candidate), tasks={})
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
    if matched_exposure and a['step'] != b['step']:
        raise RuntimeError('Unequal exposure in endpoint control comparison')
    row.update(baseline_step=a['step'], candidate_step=b['step'],
               baseline_text_tokens=a['step']*c.GLOBAL_BATCH*c.CONTEXT,
               candidate_text_tokens=b['step']*c.GLOBAL_BATCH*c.CONTEXT,
               matched_exposure=a['step'] == b['step'],
               checkpoint_selection='validation' if candidate.name in {'best', 'embedding_best'} else 'fixed_endpoint')
    passed &= b['nll'] <= a['nll']+c.NLL_ATOL
    row.update(source_nll=a['nll'], nll=b['nll'], nll_ci=paired_interval(a['per_sequence'], b['per_sequence']),
               observed_retention=bool(passed), interpretation='Point-estimate criterion; one seed; CIs are per-task, not simultaneous proof')
    return row


def build_report():
    baseline = e.RUN_ROOT/'out/smollm2_original_ar/eval/source'
    reports = []
    for exp in e.collect_experiments():
        for marker in sorted((e.output_dir(exp)/'eval').glob('*/complete.json')):
            reports.append(compare(baseline, marker.parent))
    controls = []
    staged = e.RUN_ROOT/'out/smollm2_staged_sps/eval'
    for label in ('stage_endpoint', 'final'):
        for arm in ('smollm2_direct_sps', 'smollm2_continued_ar'):
            control = e.RUN_ROOT/'out'/arm/'eval'/label
            if (staged/label/'complete.json').exists() and (control/'complete.json').exists():
                controls.append(compare(control, staged/label, matched_exposure=True))
    return dict(against_original=reports, matched_endpoint_controls=controls,
                selection_note='Validation-selected best checkpoints may have unequal exposure; use endpoint controls for warmup claims.')


def main():
    print(json.dumps(build_report(), indent=2))


if __name__ == '__main__':
    main()
