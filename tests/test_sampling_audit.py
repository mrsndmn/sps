import copy
from pathlib import Path
import pytest
from workloads.serving import sampling_audit as a


def test_probability_oracle_and_deterministic_support():
    assert a.constrained_probabilities([1,0,0],1,1,3) == [1,0,0]
    assert a.constrained_probabilities([4,2,1],1,1,1) == [1,0,0]
    p = a.constrained_probabilities([4,2,1],1,.8,3)
    assert p == pytest.approx([2/3,1/3,0])
    assert a.frequency_check([200,100],[2/3,1/3],300)['passed']
    assert not a.frequency_check([299,1],[1,0],300)['passed']


def test_cache_comparison_detects_tokens_stop_and_pairing():
    row=dict(prefix=0,limit=2,ignore_eos=False,tokens=[1,2],stop='length',sampled_logprobs=[-.5,-.2])
    groups={'sps-eager':{'samples':[row]}, 'sps_mtp-graphs':{'samples':[copy.deepcopy(row)]}}
    assert not a.compare_decodes(groups)['sps_mtp-graphs']['mismatches']
    groups['sps_mtp-graphs']['samples'][0]['tokens']=[1,3]
    assert a.compare_decodes(groups)['sps_mtp-graphs']['mismatches']
    groups['sps_mtp-graphs']['samples'][0]['limit']=3
    with pytest.raises(RuntimeError,match='pairing'):
        a.compare_decodes(groups)


def test_payload_is_single_gpu_staged_and_separate():
    body=a.payload(a.experiment_configs[0],Path('/staged/commit'))
    assert body['instance_type']==a.launch.INSTANCE_TYPES_BY_NUM_GPUS[1]
    assert 'scripts/serving/sampling_audit.py --worker' in body['script']
    assert '/staged/commit' in body['script']
    assert a.ROOT != a.source.ROOT and a.ROOT != a.graphs.ROOT


def test_completed_audit_checks_identity_before_skipping(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'ROOT',tmp_path)
    monkeypatch.setattr(a,'input_identity',lambda: {'design_signature':'new'})
    a.launch.store_json(a.output_dir()/'evaluated.json',dict(identity={'design_signature':'old'},passed=True))
    with pytest.raises(RuntimeError,match='Stale'):
        a.submit()
