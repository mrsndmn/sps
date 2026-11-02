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
