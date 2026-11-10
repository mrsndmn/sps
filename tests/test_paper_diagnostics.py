import json
import pytest
from scripts.paper_diagnostics import align, paired, read_rows, transitions


def row(i=0, answer='A', target='A'):
    return dict(index=i, doc_hash=f'hash{i}', seed=i, subject='algebra', target=target,
                answer=answer, correct=answer==target)


def test_paired_intervals_use_discordance():
    result = paired([1, 1, 0, 0], [1, 0, 1, 1])
    assert result['gained'] == 2 and result['lost'] == 1 and result['delta_pp'] == 25
    assert paired([1,0], [1,0])['ci95_pp'] == [0,0]
    with pytest.raises(ValueError):
        paired([], [])


def test_transition_decomposition_reconciles_accuracy():
    a = {i:row(i, answer=x) for i,x in enumerate([None, 'B', 'A', 'A'])}
    b = {i:row(i, answer=x) for i,x in enumerate(['A', 'A', 'B', None])}
    t = transitions(a,b,list(a))
    assert sum(t.values()) == 4
    assert t['invalid->correct'] == t['wrong->correct'] == t['correct->wrong'] == t['correct->invalid'] == 1


def test_read_rejects_incomplete_duplicate_and_inconsistent(tmp_path):
    path = tmp_path/'rows.jsonl'
    for content in [json.dumps(row()), (json.dumps(row())+'\n')*2,
                    json.dumps(dict(row(), correct=False))+'\n']:
        path.write_text(content)
        with pytest.raises(ValueError):
            read_rows(path, 'mmlu')
    path.write_text(json.dumps(row())+'\n')
    rows, evidence = read_rows(path, 'mmlu')
    assert rows[0] == row() and evidence['count'] == 1


def test_alignment_checks_identity_outside_intersection():
    a={0:row(), 1:row(1)}
    b={0:row(), 1:dict(row(1), seed=999)}
    c={0:row()}
    with pytest.raises(ValueError, match='identity'):
        align({'a':a,'b':b,'c':c}, 'mmlu')
    with pytest.raises(ValueError, match='541'):
        align({'a':a}, 'ifeval', require_full=True)
