import json
import os
import subprocess
import sys

import pytest
import torch


def test_unfreeze_registry_and_matched_continuation():
    code='''
import experiments as e,json
from scripts.mlspace_launch import payload
print(json.dumps(dict(names=[x.name for x in e.collect_experiments()],horizons=[x.horizons for x in e.collect_experiments()],
 root=str(e.ARTIFACTS_ROOT),steps=e.MTP_TRAIN_STEPS,budget=e.TRAIN_TOKENS,parent=e.MTP_PARENT_INPUT_TOKENS,
 lr=e.MTP_LR,emb_lr=e.MTP_EMBED_LR,cmd=e.train_command(e.collect_experiments()[0]),
 payload=payload('prepare','data-and-probes',e.STAGING_ROOT,'a','b'))))
'''
    d=json.loads(subprocess.check_output([sys.executable,'-c',code],text=True,env=dict(os.environ,SPS_EXPERIMENT_SUITE='mtp-unfreeze')))
    assert d['horizons']==[2,3,4]
    assert d['root'].endswith('/mtp-unfreeze')
    assert d['parent']==1_000_013_824
    assert d['steps']==763 and d['budget']==50_003_968
    assert d['lr']==1e-5 and d['emb_lr']==1e-4
    assert 'scripts.mtp_train' in d['cmd']
    assert 'scripts.mtp_run' in d['payload']['script']
    assert d['payload']['instance_type']=='a100.4gpu'


def test_plain_ce_includes_first_horizon_and_padding():
    from scripts.mtp_train import adaptation_loss
    torch.manual_seed(7)
    logits=torch.randn(2,3,4,9,requires_grad=True)
    labels=torch.randint(0,9,(2,3,4));labels[0,1,2:]=-100
    loss=adaptation_loss(logits,labels,all_horizons=True)
    expected=torch.nn.functional.cross_entropy(logits.reshape(-1,9),labels.reshape(-1))
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert logits.grad[:,:,0].abs().sum()>0
    assert logits.grad[0,1,2:].abs().sum()==0


def test_optimizer_deduplicates_tied_embeddings_and_preserves_lr_ratio(monkeypatch):
    from scripts.mtp_train import optimizer_groups
    import experiments as e
    from types import SimpleNamespace
    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.reference=torch.nn.Module()
            self.reference.transformer=torch.nn.Module()
            self.reference.transformer.wte=torch.nn.Embedding(7,4)
            self.reference.lm_head=torch.nn.Linear(4,7,bias=False)
            self.reference.lm_head.weight=self.reference.transformer.wte.weight
            self.reference.block=torch.nn.Linear(4,4)
            self.new_embeddings=torch.nn.Parameter(torch.ones(2,4))
    monkeypatch.setattr(e,'MTP_UNFREEZE',True)
    monkeypatch.setattr(e,'MTP_LR',1e-5)
    monkeypatch.setattr(e,'MTP_EMBED_LR',1e-4)
    model=Draft();groups=optimizer_groups(model)
    params=[p for g in groups for p in g['params']]
    assert len(params)==len({id(p) for p in params})==len(list(model.parameters()))
    tied=model.reference.transformer.wte.weight
    for g in groups:
        if any(p is tied for p in g['params']):
            assert g['lr']==1e-4 and g['weight_decay']==0
        if any(p is model.reference.block.weight for p in g['params']):assert g['lr']==1e-5
        if any(p is model.reference.block.bias for p in g['params']):assert g['weight_decay']==0


def test_full_checkpoint_restores_backbone_and_tied_weights(tmp_path,monkeypatch):
    import experiments as e
    from scripts.mtp_train import save_checkpoint
    from scripts.mtp_common import restore_draft
    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__();self.horizons=2
            self.new_embeddings=torch.nn.Parameter(torch.randn(1,4))
            self.reference=torch.nn.Module();self.reference.transformer=torch.nn.Module()
            self.reference.transformer.wte=torch.nn.Embedding(9,4)
            self.reference.lm_head=torch.nn.Linear(4,9,bias=False)
            self.reference.lm_head.weight=self.reference.transformer.wte.weight
    monkeypatch.setattr(e,'MTP_UNFREEZE',True)
    model=Draft();optimizer=torch.optim.AdamW(model.parameters())
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    path=tmp_path/'full.pt';save_checkpoint(path,model,optimizer,3,'sig')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['train_backbone'] and 'backbone' in saved
    with torch.no_grad():
        for p in model.parameters():p.add_(10)
    restore_draft(model,saved)
    for n,p in model.named_parameters():torch.testing.assert_close(p,before[n],rtol=0,atol=0)
    assert model.reference.lm_head.weight is model.reference.transformer.wte.weight


def test_changed_parent_cannot_be_used_for_before_after_evaluation(tmp_path,monkeypatch):
    from scripts import mtp_common
    source=tmp_path/'source.pt';source.write_bytes(b'source')
    data=tmp_path/'data';data.mkdir();(data/'manifest.json').write_bytes(b'data')
    parent=tmp_path/'parent.pt';parent.write_bytes(b'learned embeddings')
    trained=dict(source_sha256=mtp_common.file_sha256(source),data_manifest_sha256=mtp_common.file_sha256(data/'manifest.json'),
        warmstart=dict(path=str(parent),sha256=mtp_common.file_sha256(parent)))
    monkeypatch.setattr(mtp_common,'MTP_SOURCE',source);monkeypatch.setattr(mtp_common,'MTP_DATA',data)
    parent.write_bytes(b'other learned embeddings')
    with pytest.raises(RuntimeError,match='Warmstart.*changed'):
        mtp_common.validate_evaluation_inputs(trained)
