import pytest
import torch
from modeling.models.sps.mtp import SPSMTP
from modeling.models.sps.unified import UnifiedSPS,unified_block
from test_mtp_evaluate import model as tiny


@pytest.mark.parametrize('horizons',[2,3,4])
@pytest.mark.parametrize('tokens',[[4,5,6],[4,16,5]])
def test_fused_groups_match_independent_full_prefix_oracle(horizons,tokens):
    m=SPSMTP(tiny().reference,horizons,backend='sdpa').eval()
    m.new_embeddings.data.add_(torch.randn_like(m.new_embeddings)*.1)
    prefix=torch.tensor([1,2,3]);t=torch.tensor(tokens)
    cache,_=m.encode(prefix[None],positions=[2])
    state,got=unified_block(m,cache,t)
    for j in range(t.numel()):
        ids=torch.cat((prefix,t[:j+1]))
        _,want=m.encode(ids[None],positions=[ids.numel()-1])
        torch.testing.assert_close(got[j],want[0,0],atol=3e-5,rtol=3e-4)
    assert state.length==prefix.numel()+t.numel()
    changed=t.clone();changed[-1]=(changed[-1]+1)%16
    _,other=unified_block(m,cache,changed)
    torch.testing.assert_close(got[:-1],other[:-1],atol=0,rtol=0)


@pytest.mark.parametrize('accepted',[0,1,2])
def test_rejection_rolls_back_all_groups_and_reuses_verified_work(accepted):
    m=SPSMTP(tiny().reference,3,backend='sdpa').eval();v=UnifiedSPS(m)
    prefix=torch.tensor([1,2]);v.ensure(prefix)
    history=torch.tensor([1,2,3]);draft=torch.tensor([4,5]);v.verify_logits(history,draft)
    current=torch.cat((history,draft[:accepted],torch.tensor([9])))
    q=v.propose(current)
    _,want=m.encode(current[:-1][None],positions=[current.numel()-2])
    torch.testing.assert_close(q,want[0,0,1:],atol=3e-5,rtol=3e-4)
    assert v.block_calls==1 and v.prefills==1
    v.verify_logits(current,torch.tensor([6,7]))
    assert v.block_calls==2 and v.prefills==1


@pytest.mark.parametrize('budget',[0,1,2,3,7])
@pytest.mark.parametrize('horizons',[2,3,4])
def test_unified_greedy_matches_same_checkpoint_p1(budget,horizons):
    from scripts.mtp_unified_evaluate import generate
    from scripts.mtp_evaluate import probabilities
    m=SPSMTP(tiny().reference,horizons,backend='sdpa').eval();prefix=torch.tensor([1,2])
    ids,result,engine,_,_=generate(m,prefix,budget,'greedy',1337)
    history=prefix;want=[]
    for _ in range(budget):
        _,logits=m.encode(history[None],positions=[history.numel()-1])
        token=probabilities(logits[0,0,0],m.config).argmax();want.append(int(token));history=torch.cat((history,token[None]))
        if int(token)==m.config.eos_token_id:break
    assert ids.tolist()==want
    assert engine.block_calls==(result.rounds if result else 0)


def test_unified_suite_is_evaluation_only_and_uses_continuation_markers():
    import subprocess,os
    r=subprocess.run(['python','run_eval.py','--dry'],env={**os.environ,'SPS_EXPERIMENT_SUITE':'mtp-unified'},capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert r.stdout.count('scripts.mtp_unified_evaluate')==3
    assert 'a100.4gpu' in r.stdout
    r=subprocess.run(['python','run_train_jobs.py','--phase','train','--dry'],env={**os.environ,'SPS_EXPERIMENT_SUITE':'mtp-unified'},capture_output=True,text=True)
    assert r.returncode!=0 and 'evaluation-only' in r.stderr


def test_sampling_anchor_and_residual_match_self_target(monkeypatch):
    from types import SimpleNamespace
    import scripts.mtp_unified_evaluate as ev
    target=torch.tensor([[.75,.25,0.,0.,0.],[.2,.8,0.,0.,0.]])
    config=SimpleNamespace(eos_token_id=2,pad_token_id=3,predict_token_id=4)
    class Engine:
        def __init__(self,model):self.prefills=1;self.block_calls=0
        def ensure(self,prefix):return torch.stack((target[int(prefix[-1])].log(),target[0].log()))
        def propose(self,history):return torch.tensor([[.05,.95,0.,0.,0.]]).log()
        def verify_logits(self,history,draft):
            self.block_calls+=1
            return torch.stack((target[int(history[-1])],target[int(draft[0])])).log()
    monkeypatch.setattr(ev,'UnifiedSPS',Engine)
    counts=torch.zeros(2,2)
    for seed in range(2500):
        ids,*_=ev.generate(SimpleNamespace(config=config),torch.tensor([0]),2,'sample',seed)
        counts[ids[0],ids[1]]+=1
    expected=target[0,:2,None]*target[:,:2]
    torch.testing.assert_close(counts/2500,expected,atol=.025,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA fused attention/native comparison')
def test_fused_cuda_matches_full_horizons_and_native_p1():
    from modeling.models.sps import SPSModel,SPSConfig
    from scripts.mtp_evaluate import probabilities
    from modeling.models.sps.parallel_verify import ParallelPrefixVerifier
    ref=SPSModel(SPSConfig(vocab_size=39,block_size=32,n_layer=2,n_head=2,
        hidden_size=128,intermediate_size=192,eos_token_id=36,pad_token_id=37,
        predict_token_id=38,window_size=0,dropout=0.)).cuda().eval()
    m=SPSMTP(ref,4,backend='sdpa').eval();m.new_embeddings.data.normal_(std=.1)
    prefix=torch.tensor([1,2,3],device='cuda');draft=torch.tensor([4,36,5],device='cuda')
    engine=UnifiedSPS(m);engine.ensure(prefix)
    _,got=unified_block(m,engine.cache,draft)
    ids=torch.cat((prefix,draft))
    _,want=m.encode(ids[None],positions=torch.arange(3,6,device='cuda'))
    torch.testing.assert_close(probabilities(got,m.config),probabilities(want[0],m.config),atol=1e-3,rtol=1e-2)
    native=ParallelPrefixVerifier(ref)
    expected=native.verify_logits(prefix,draft)[1:]
    torch.testing.assert_close(probabilities(got[:,0],m.config),probabilities(expected,m.config),atol=1e-3,rtol=1e-2)
