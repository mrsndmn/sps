import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts import sampling_trace as t


def test_extract_pairs_deduplicates_first_token_divergence():
    prefixes=[{'tokens':[10,11]},{'tokens':[20]}]
    a=[dict(prefix=0,limit=128,ignore_eos=True,tokens=[1,2]),dict(prefix=1,limit=128,ignore_eos=True,tokens=[3,4])]
    b=[dict(a[0],tokens=[2,2]),dict(a[1],tokens=[3,5])]
    cases=t.extract_cases(prefixes,a,b)
    assert len(cases)==3 and cases[0]['plain_token']==1
    assert cases[-1]['tokens']==[20,3] and cases[-1]['offset']==1
    with pytest.raises(ValueError):t.extract_cases(prefixes,a,b[:1])


def test_distribution_comparison_distinguishes_near_tie_from_drift():
    result=t.compare(np.log([.501,.499]),np.log([.499,.501]))
    assert result['argmax_a']!=result['argmax_b']
    assert result['total_variation']==pytest.approx(.002)
    with pytest.raises(ValueError):t.compare(np.log([.2,.2]),np.log([.5,.5]))


def test_four_slot_intervention_preserves_eager_no_speculation_contract():
    import torch
    model=SimpleNamespace(sps_horizons=1,config=SimpleNamespace(sps_horizons=4,hidden_size=8),sps_hidden=torch.empty(16,1,8))
    runner=SimpleNamespace(model=model,vllm_config=SimpleNamespace(model_config=SimpleNamespace(enforce_eager=True),speculative_config=None))
    worker=SimpleNamespace(model_runner=runner)
    assert t.TraceWorkerExtension.sps_trace_four_slots(worker)['horizons']==4
    assert model.sps_hidden.shape==(16,4,8)
    runner.vllm_config.model_config.enforce_eager=False
    with pytest.raises(RuntimeError):t.TraceWorkerExtension.sps_trace_four_slots(worker)


def test_submission_adapter_does_not_recurse_identity_or_touch_parent(tmp_path,monkeypatch):
    base={'checkpoint':'same','design_signature':'archived'}
    live=dict(base, design_signature='current')
    monkeypatch.setattr(t.audit,'live_input_identity',lambda:live)
    monkeypatch.setattr(t,'SOURCE',tmp_path)
    for name in t.FILES:
        (tmp_path/name).write_text('{}')
    (tmp_path/'identity.json').write_text(json.dumps(base))
    (tmp_path/'report.json').write_text(json.dumps({'identity':base}))
    expected=t.identity()
    assert expected['audit'] == base
    with t.patch.object(t.audit,'input_identity',t.identity):
        assert t.audit.input_identity()==expected
    live['checkpoint']='changed'
    with pytest.raises(RuntimeError,match='different runtime or checkpoint'):
        t.identity()
    body=t.payload(t.experiment_configs[0],Path('/staged'))
    assert body['instance_type']==t.launch.INSTANCE_TYPES_BY_NUM_GPUS[1]
    assert 'scripts.sampling_trace --worker' in body['script']
    assert t.ROOT!=t.audit.ROOT


def test_ties_use_first_token_and_are_reported_explicitly():
    result=t.compare(np.log([.5,.5]),np.log([.49,.51]))
    assert result['argmax_a']==0 and result['argmax_b']==1
    assert result['top_ties_a']==2 and result['top_ties_b']==1
    assert result['top_gap_a']==0
