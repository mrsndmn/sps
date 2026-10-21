import json
import os
import subprocess
import sys

import torch


def test_mtp_registry_isolated_and_four_gpu_dry():
    script = '''
import json
import experiments as e
from scripts.mlspace_launch import payload
print(json.dumps(dict(root=str(e.ARTIFACTS_ROOT), names=[x.name for x in e.collect_experiments()],
 horizons=[x.horizons for x in e.collect_experiments()],budget=e.TRAIN_TOKENS,
 command=e.train_command(e.collect_experiments()[0]),
 payload=payload('train',e.collect_experiments()[0].name,e.STAGING_ROOT,'a','b'))))
'''
    d=json.loads(subprocess.check_output([sys.executable,'-c',script],env=dict(os.environ,SPS_EXPERIMENT_SUITE='mtp'),text=True))
    assert d['horizons']==[2,3,4]
    assert d['root'].endswith('/sps-artifacts/mtp')
    assert d['budget']>=1_000_000_000
    assert d['payload']['instance_type']=='a100.4gpu'
    assert 'scripts.mtp_run' in d['payload']['script']
    assert d['payload']['env_variables']['SPS_EXPERIMENT_SUITE']=='mtp'
    assert 'scripts.mtp_train' in d['command']


def test_targets_never_cross_eos_or_padding():
    from scripts.mtp_common import shifted_targets
    x=torch.tensor([[3,4,9,5,6,7,8]])
    targets=shifted_targets(x,4,3,eos=9,pad=0)
    assert targets.tolist()==[[[4,9,-100],[9,-100,-100],[-100,-100,-100],[6,7,8]]]
    targets=shifted_targets(torch.tensor([[3,4,0,0]]),2,2,eos=9,pad=0)
    assert targets.tolist()==[[[4,-100],[-100,-100]]]


def test_matching_prefixes_not_changed_by_horizon_count():
    from scripts.mtp_common import sequence_starts
    a=sequence_starts(30000,1024,4,0,seed=1337)
    b=sequence_starts(30000,1024,4,0,seed=1337)
    assert list(a)==list(b)
    starts=[list(sequence_starts(30000,1024,4,r,seed=1337)) for r in range(4)]
    flat=[x for row in starts for x in row]
    assert len(flat)==len(set(flat))


def test_new_horizon_loss_ignores_original_predict_and_masked_targets():
    from scripts.mtp_train import adaptation_loss
    logits=torch.zeros(1,2,3,5,requires_grad=True)
    labels=torch.tensor([[[1,2,3],[2,-100,4]]])
    loss=adaptation_loss(logits,labels)
    assert abs(float(loss.detach())-__import__('math').log(5))<1e-6
    loss.backward()
    assert logits.grad[:,:,0].abs().sum()==0
    assert logits.grad[0,1,1].abs().sum()==0
    assert logits.grad[:,:,2].abs().sum()>0


def test_mtp_forecast_excludes_compilation_and_enforces_full_budget(tmp_path,monkeypatch):
    import experiments as e
    from scripts import mtp_run
    monkeypatch.setattr(e,'SANITY_ROOT',tmp_path)
    monkeypatch.setattr(e,'MTP_TRAIN_STEPS',100)
    exp=e.ExperimentConfig('mtp','sps',horizons=4)
    p=tmp_path/'out/mtp/metrics.jsonl';p.parent.mkdir(parents=True)
    p.write_text('\n'.join(json.dumps(dict(event='train',step_seconds=x)) for x in [10000,10000]+[2]*8))
    estimate=mtp_run.estimate(exp)
    assert estimate['seconds_per_step']==2
    assert estimate['estimated_hours_range'][0]<estimate['estimated_hours_range'][1]<2
    assert estimate['token_budget']==e.TRAIN_TOKENS
