from contextlib import nullcontext
import json
from pathlib import Path
import pytest
import torch
from scripts import provenance_evaluate as p


def test_complete_registry_and_one_gpu_payloads():
    assert {(e.arm,e.mode) for e in p.experiment_configs} == {
        (e.name,m) for e in p.train.experiment_configs for m in ('thinking','nonthinking')}
    for exp in p.experiment_configs:
        body=p.payload(exp,Path('/staged'))
        assert body['instance_type']==p.launch.INSTANCE_TYPES_BY_NUM_GPUS[1]
        assert f'--worker --filter {exp.name}' in body['script']
    assert p.ROOT != p.train.ROOT


def test_pinned_identity_fails_closed(tmp_path):
    path=tmp_path/'input.json'
    ident=dict(checkpoint={'sha256':'x'},protocol={'seed':1})
    with pytest.raises(RuntimeError):p.require_identity(path,ident)
    p.launch.store_json(path,ident)
    p.require_identity(path,ident)
    with pytest.raises(RuntimeError):p.require_identity(path,dict(ident,checkpoint={'sha256':'other'}))


def test_evaluation_rejects_multiple_gpus_before_loading(monkeypatch):
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    with pytest.raises(RuntimeError,match='exactly one GPU'):
        p.evaluate(p.experiment_configs[0])


def test_checkpoint_uses_fixed_final_and_training_identity(tmp_path,monkeypatch):
    p.checkpoint.cache_clear()
    exp=p.train.experiment_configs[0]
    ident={'plan':{'steps':461}}
    (tmp_path/'final.pt').write_bytes(b'weights')
    marker=dict(identity=ident,passed=True,checkpoint=str(tmp_path/'final.pt'),checkpoint_sha256='recorded')
    p.launch.store_json(tmp_path/'trained.json',marker)
    monkeypatch.setattr(p.train,'output_dir',lambda exp:tmp_path)
    monkeypatch.setattr(p.train,'paired_data',lambda:({},{}))
    monkeypatch.setattr(p.train,'make_plan',lambda _:ident['plan'])
    monkeypatch.setattr(p.train,'identity',lambda *args:ident)
    monkeypatch.setattr(p.train,'training_context',lambda *args:nullcontext())
    checked=[]
    monkeypatch.setattr(p.train.common,'validate_final',lambda path,plan:checked.append((path,plan)))
    cp=p.checkpoint(exp.name)
    assert cp['path']==str(tmp_path/'final.pt') and cp['sha256']=='recorded'
    assert checked==[(tmp_path/'final.pt',ident['plan'])]
    p.checkpoint.cache_clear()
    marker['checkpoint']=str(tmp_path/'best.pt')
    p.launch.store_json(tmp_path/'trained.json',marker)
    with pytest.raises(RuntimeError,match='identity'):p.checkpoint(exp.name)
    p.checkpoint.cache_clear()
