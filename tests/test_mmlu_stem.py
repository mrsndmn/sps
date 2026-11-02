import hashlib
import json
import pytest
from scripts.report_mmlu_stem import STEM_SUBJECTS, read_samples, summarize


def sample(index, row, answer):
    return dict(index=index, doc_hash=hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
                subject=row['subject'], target='ABCD'[row['answer']], answer=answer, stop='answer')


def test_stem_selection_and_weighted_accuracy():
    data = [dict(subject=s, answer=0) for s in ['astronomy', 'astronomy', 'machine_learning', 'anatomy']]
    rows = [sample(i, x, a) for i, (x, a) in enumerate(zip(data, ['A', 'A', 'B', 'A']))]
    result = summarize(rows[:3], data)
    assert result['complete'] and result['expected'] == 3
    assert result['accuracy'] == pytest.approx(2/3)
    assert result['macro_accuracy'] == .5
    assert summarize(rows[:2], data)['accuracy'] is None
    assert len(STEM_SUBJECTS) == 18 and 'anatomy' not in STEM_SUBJECTS
    with pytest.raises(ValueError, match='Duplicate'):
        summarize(rows + rows[:1], data)
    rows[0]['doc_hash'] = 'wrong'
    with pytest.raises(ValueError, match='provenance'):
        summarize(rows, data)


def test_active_file_is_never_modified(tmp_path):
    p = tmp_path / 'samples.jsonl'
    content = b'{"index":0}\n{"index":1}'
    p.write_bytes(content)
    assert read_samples(p) == [{'index': 0}]
    assert p.read_bytes() == content


def test_subset_reuse_noncontiguous_ids(tmp_path):
    from types import SimpleNamespace
    from scripts.mmlu_subset import selected_indices, reuse_samples, validate_saved, SOURCE_SIGNATURE
    data = [dict(subject=s, answer=0) for s in ['anatomy', 'astronomy', 'anatomy', 'machine_learning']]
    indices = selected_indices(data, stem=True)
    assert indices == [1, 3]
    assert selected_indices(data, stem=True, limit=1) == [1]
    exp = SimpleNamespace(name='fixture')
    (tmp_path/'manifest.json').write_text(json.dumps(dict(design_signature=SOURCE_SIGNATURE, limit=None, experiment=exp.__dict__)))
    rows = [sample(i, row, 'A') | {'correct': True} for i, row in enumerate(data)]
    (tmp_path/'samples.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    reused = reuse_samples(tmp_path, exp, data, indices)
    assert [x['index'] for x in reused] == indices
    assert validate_saved(list(reversed(reused)), data, indices) == {1, 3}
    with pytest.raises(ValueError, match='Unexpected'):
        validate_saved(rows, data, indices)
    (tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='manifest'):
        reuse_samples(tmp_path, exp, data, indices)


def test_merge_full_stem_deduplicates_and_rejects_conflicts(tmp_path):
    from types import SimpleNamespace
    from scripts.mmlu_subset import reuse_full_and_stem, SOURCE_SIGNATURE, STEM_SIGNATURE
    data = [dict(subject=s, answer=0) for s in ['anatomy', 'astronomy', 'machine_learning']]
    exp = SimpleNamespace(name='fixture')
    rows = [sample(i, row, 'A') | {'correct': True} for i, row in enumerate(data)]
    for suite, signature, selected in [('mmlu-generation', SOURCE_SIGNATURE, rows[:2]),
                                       ('mmlu-stem-generation', STEM_SIGNATURE, rows[1:])]:
        folder = tmp_path/suite/'runs/out'/exp.name/'full'
        folder.mkdir(parents=True)
        (folder/'manifest.json').write_text(json.dumps(dict(design_signature=signature, limit=None, experiment=exp.__dict__)))
        (folder/'samples.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in selected))
    assert reuse_full_and_stem(tmp_path, exp, data, [0, 1, 2]) == rows
    rows[1]['answer'] = 'B'; rows[1]['correct'] = False
    (folder/'samples.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows[1:]))
    with pytest.raises(ValueError, match='Conflicting'):
        reuse_full_and_stem(tmp_path, exp, data, [0, 1, 2])
