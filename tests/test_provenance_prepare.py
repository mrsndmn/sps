import copy
from pathlib import Path
import pytest
from scripts import provenance_prepare as p


def row(mode, i):
    return dict(mode=mode,source_id=str(i)*64,group_id=str(i)*64,
        messages=[{'role':'user','content':'prompt'},{'role':'assistant','content':'response'}],
        input_ids=[1,2,3],loss_mask=[0,1,1])


def record(old, new=None, rejected=None):
    result=dict(design_signature='sig',source_id=old['source_id'],source_row_sha256=p.data.digest(old),split='train')
    if rejected:result['rejected']=rejected
    else:result['row']=new or copy.deepcopy(old)
    return result


def test_pairs_preserve_thinking_and_remove_same_rejections(monkeypatch):
    rows=[row('thinking',0),row('nonthinking',1),row('nonthinking',2)]
    updated=copy.deepcopy(rows[1]);updated['messages'][-1]['content']='teacher'
    samples={rows[1]['source_id']:record(rows[1],updated),rows[2]['source_id']:record(rows[2],rejected='incomplete')}
    monkeypatch.setattr(p,'MIN_PAIRED_FRACTION',.5)
    a,b,rejected=p.pair_rows(rows,samples,'train','sig')
    assert a[0] is b[0] is rows[0]
    assert [r['source_id'] for r in a]==[r['source_id'] for r in b]
    assert a[1]['messages'][-1]['content']=='response'
    assert b[1]['messages'][-1]['content']=='teacher'
    assert rejected=={'incomplete':1}


def test_pairing_rejects_changed_prompt_and_stale_source():
    old=row('nonthinking',1);new=copy.deepcopy(old)
    new['messages'][0]['content']='different prompt'
    with pytest.raises(RuntimeError,match='prompt identity'):
        p.validate_record(record(old,new),old,'train','sig')
    stale=record(old);stale['source_row_sha256']='stale'
    with pytest.raises(RuntimeError,match='identity'):
        p.validate_record(stale,old,'train','sig')


def test_rejection_floor_prevents_biased_tiny_dataset():
    rows=[row('thinking',0),row('nonthinking',1)]
    with pytest.raises(RuntimeError,match='Insufficient'):
        p.pair_rows(rows,{rows[1]['source_id']:record(rows[1],rejected='incomplete')},'train','sig')


def test_preparation_gpu_allocation_and_budget_summary():
    body=p.payload(p.experiment_configs[0],Path('/staged/commit'))
    assert body['instance_type']==p.launch.INSTANCE_TYPES_BY_NUM_GPUS[4]
    assert 'scripts.provenance_prepare --worker' in body['script']
    result=p.summarize([row('thinking',0),row('nonthinking',1)])
    assert result['thinking']['supervised_tokens']==2
    assert result['nonthinking']['examples']==1
