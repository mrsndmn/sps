import pytest
import torch
from modeling.spec_baselines import MedusaHeads, TinyAR, lookup_proposal


def test_lookup_uses_only_supplied_history():
    p=torch.tensor([1,2,3,4,1,2])
    d,q=lookup_proposal(p,3,10,max_match=4)
    assert d[:2].tolist()==[3,4]
    assert torch.equal(q.argmax(-1),d)
    assert torch.all(q.sum(-1)==1)


def test_medusa_heads_gradient_without_changing_features_or_head():
    head=torch.nn.Linear(8,13,bias=False).requires_grad_(False)
    model=MedusaHeads(8,3)
    features=torch.randn(2,4,8)
    logits=model(features,head)
    assert logits.shape==(2,4,3,13)
    logits.square().mean().backward()
    assert all(p.grad is not None for p in model.parameters())
    assert head.weight.grad is None and features.grad is None


def tiny_config():
    from types import SimpleNamespace
    return SimpleNamespace(vocab_size=19,hidden_size=16,n_head=2,n_layer=2,
        intermediate_size=24,norm_eps=1e-6,bias=False,dropout=0.,block_size=32,
        eos_token_id=16,pad_token_id=17,predict_token_id=18)


def test_ar_causality_cache_and_rejection():
    from modeling.models.sps.mtp import truncate_cache
    torch.manual_seed(2);model=TinyAR(tiny_config()).eval()
    p=torch.tensor([[1,2,16,3,4,5]])
    with torch.no_grad():
        full=model(p)
        altered=p.clone();altered[0,-1]=7
        torch.testing.assert_close(full[:,:-1],model(altered)[:,:-1],atol=0,rtol=0)
        cache=None;rows=[]
        for i in range(p.shape[1]):
            cache,logits=model.encode(p[:,i:i+1],cache);rows.append(logits)
        torch.testing.assert_close(full,torch.cat(rows,1),atol=1e-6,rtol=1e-5)
        cache=truncate_cache(cache,4)
        cache,got=model.encode(torch.tensor([[9]]),cache,positions=[4])
        torch.testing.assert_close(got,model(torch.tensor([[1,2,16,3,9]]))[:,-1:],atol=1e-6,rtol=1e-5)
    assert model.lm_head.weight is model.wte.weight


def test_medusa_training_freezes_source_and_masks_future_gaps(monkeypatch):
    import experiments as e
    from scripts.spec_baseline_common import MedusaTraining,loss_for
    from test_mtp_evaluate import model as source
    monkeypatch.setattr(e,'BASELINE_CONTEXT',4)
    m=MedusaTraining(source().reference).train()
    ids=torch.tensor([[1,2,3,4,5,6,7,8,9]])
    logits=m(ids[:,:4]);loss=loss_for(logits,ids,'medusa',m.config);loss.backward()
    assert not m.encoder.reference.training
    assert not any(p.requires_grad for p in m.encoder.parameters())
    assert all(p.grad is None for p in m.encoder.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.heads.parameters())


def test_medusa_generation_uses_verifier_features_without_extra_trunk_pass(monkeypatch):
    from test_mtp_evaluate import model as source
    import modeling.models.sps.parallel_verify as pv
    from scripts.spec_baseline_evaluate import Proposal
    ref=source().reference;v=pv.ParallelPrefixVerifier(ref)
    p=torch.tensor([1,2,3]);v.ensure(p)
    heads=MedusaHeads(ref.config.hidden_size,4)
    proposal=Proposal('medusa',heads,ref,v,3)
    count=[];original=pv.block_decode
    def spy(*args,**kwargs):count.append(args[2].numel());return original(*args,**kwargs)
    monkeypatch.setattr(pv,'block_decode',spy)
    history=torch.cat((p,torch.tensor([4])))
    with torch.inference_mode():
        q=proposal.independent(history)
        assert q.shape==(3,ref.config.vocab_size) and count==[]
        draft=q.argmax(-1);v.verify_logits(history,draft)
        assert count==[4]
        next_history=torch.cat((history,draft[:1],torch.tensor([9])))
        proposal.independent(next_history)
        assert count==[4]
        # The selected hidden state must reproduce that exact cached prefix's logits.
        features=v.ensure_features(next_history[:-1])
        torch.testing.assert_close(ref.lm_head(features),v.logits[-1],atol=1e-6,rtol=1e-5)


def test_ar_rows_condition_on_actual_previous_draft_tokens():
    from scripts.spec_baseline_evaluate import Proposal
    from scripts.mtp_evaluate import probabilities
    m=TinyAR(tiny_config()).eval();p=torch.tensor([1,2,3])
    proposal=Proposal('ar_tiny',m,m,None,4)
    with torch.inference_mode():
        draft,q=proposal.conditional(p,3,'sample',torch.Generator().manual_seed(3))
        for j in range(3):
            ids=torch.cat((p,draft[:j]))
            expected=probabilities(m(ids[None])[0,-1,0],m.config)
            torch.testing.assert_close(q[j],expected,atol=1e-6,rtol=1e-5)
            assert q[j,draft[j]]>0
        assert q[3,0]==1 and draft[3]==0


def test_reached_counts_exclude_slots_after_rejection():
    from modeling.mtp_speculative import speculative_generate
    p=torch.tensor([0]);q=torch.tensor([[1.,0.]]).repeat(3,1)
    def verify(prefix,draft):return torch.tensor([[0.,1.]]).repeat(len(draft)+1,1)
    result=speculative_generate(p,lambda p:q,verify,max_new_tokens=3)
    assert result.token_ids.tolist()==[1,1,1]
    assert result.proposed_by_horizon==(3,2,1)
    assert result.reached_by_horizon==(3,0,0)
    assert result.accepted_by_horizon==(0,0,0)


def test_conditional_sampling_has_target_markov_law():
    from modeling.mtp_speculative import speculative_generate
    def target(last):return torch.tensor([.4,.6]) if int(last)==0 else torch.tensor([.7,.3])
    def conditional(prefix,remaining,mode,generator):
        q=[];draft=[];history=prefix
        for j in range(2):
            row=torch.tensor([.8,.2]) if int(history[-1])==0 else torch.tensor([.1,.9])
            token=torch.multinomial(row,1,generator=generator)[0]
            q.append(row);draft.append(token);history=torch.cat((history,token[None]))
        return torch.stack(draft),torch.stack(q)
    def verify(prefix,draft):return torch.stack([target(torch.cat((prefix,draft[:j]))[-1]) for j in range(len(draft)+1)])
    counts=torch.zeros(4);g=torch.Generator().manual_seed(123)
    for _ in range(2500):
        r=speculative_generate(torch.tensor([0]),lambda p:None,verify,max_new_tokens=2,
            mode='sample',generator=g,draft_proposal=conditional)
        counts[int(r.token_ids[0])*2+int(r.token_ids[1])]+=1
    torch.testing.assert_close(counts/2500,torch.tensor([.16,.24,.42,.18]),atol=.035,rtol=0)


def test_baseline_registry_dry_and_training_budget():
    import subprocess,sys,os,json
    env=dict(os.environ,SPS_EXPERIMENT_SUITE='spec-baselines')
    code="import experiments as e,json;print(json.dumps(dict(root=str(e.ARTIFACTS_ROOT),training=[x.recipe for x in e.experiment_configs if x.trainable],budget=e.TRAIN_TOKENS,commands=[e.train_command(x) for x in e.experiment_configs if x.trainable])))"
    a=json.loads(subprocess.check_output([sys.executable,'-c',code],env=env,text=True))
    assert a['root'].endswith('/spec-baselines') and a['training']==['medusa','ar_tiny']
    assert a['budget']==1050017792
    assert all('scripts.spec_baseline_train' in row for row in a['commands'])
    for command in (['run_train_jobs.py','--phase','prepare','--dry'],['run_train_jobs.py','--dry'],['run_eval.py','--dry']):
        r=subprocess.run([sys.executable,*command],env=env,capture_output=True,text=True)
        assert r.returncode==0,r.stderr
        assert 'scripts.spec_baseline_run' in r.stdout


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA mixed precision gradients')
def test_cuda_head_and_ar_training_step(monkeypatch):
    import experiments as e
    from scripts.spec_baseline_common import MedusaTraining,loss_for
    from modeling.models.sps import SPSModel,SPSConfig
    from types import SimpleNamespace
    monkeypatch.setattr(e,'BASELINE_CONTEXT',8)
    torch.manual_seed(4)
    ref=SPSModel(SPSConfig(vocab_size=39,block_size=32,n_layer=2,n_head=2,
        hidden_size=128,intermediate_size=192,eos_token_id=36,pad_token_id=37,
        predict_token_id=38,window_size=0,dropout=0.)).cuda()
    original=ref.lm_head.weight.detach().clone()
    medusa=MedusaTraining(ref).cuda().train()
    cfg=SimpleNamespace(**vars(tiny_config()));cfg.n_head=2;cfg.hidden_size=32
    tiny=TinyAR(cfg).cuda().train()
    ids=torch.tensor([[1,2,3,4,5,6,7,8,9,10,11,12,13]],device='cuda')
    for model,recipe in [(medusa,'medusa'),(tiny,'ar_tiny')]:
        parameters=[p for p in model.parameters() if p.requires_grad]
        opt=torch.optim.AdamW(parameters,lr=1e-3)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=loss_for(model(ids[:,:8]),ids,recipe,model.config)
        loss.backward()
        assert torch.isfinite(loss)
        invalid=[name for name,p in model.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
        assert not invalid,(recipe,invalid)
        opt.step()
    torch.testing.assert_close(ref.lm_head.weight,original,atol=0,rtol=0)


def test_probe_forecast_uses_only_steady_steps(tmp_path,monkeypatch):
    import experiments as e,json
    from scripts.spec_baseline_run import estimate
    monkeypatch.setattr(e,'SANITY_ROOT',tmp_path)
    exp=e.ExperimentConfig('baseline','medusa')
    dest=tmp_path/'out'/exp.name;dest.mkdir(parents=True)
    (dest/'metrics.jsonl').write_text('\n'.join(json.dumps(dict(event='train',step_seconds=t)) for t in [9999]*8+[2.]*8))
    result=estimate(exp)
    assert result['seconds_per_step']==2.
    assert result['tokens']==e.BASELINE_STEPS*e.BASELINE_STEP_TOKENS
    assert result['estimated_hours_range'][0]<result['estimated_hours_range'][1]<24


def test_conditional_acceptance_summary_uses_reached_counts():
    from scripts.spec_baseline_evaluate import summarize
    rows=[dict(tokens=3,initial_target_tokens=1,total_seconds=1.,rounds=1,
        accepted_by_horizon=[1,1,0],proposed_by_horizon=[1,1,1],reached_by_horizon=[1,1,1],
        proposal_stream_seconds=.2,verification_stream_seconds=.4,token_ids=[1,2,3]),
        dict(tokens=2,initial_target_tokens=1,total_seconds=1.,rounds=1,
        accepted_by_horizon=[0,0,0],proposed_by_horizon=[1,1,1],reached_by_horizon=[1,0,0],
        proposal_stream_seconds=.2,verification_stream_seconds=.4,token_ids=[1,4])]
    base=[dict(tokens=3,total_seconds=1.,token_ids=[1,2,3]),dict(tokens=2,total_seconds=1.,token_ids=[1,4])]
    a=summarize(rows,base,3,'greedy')
    assert a['unconditional_acceptance']==[.5,.5,0.]
    assert a['conditional_acceptance']==[.5,1.,0.]
    assert a['emitted_per_round']==1.5 and a['native_exact_prompts']==2


@pytest.mark.parametrize('change',[{'tokens':0},{'step':-1},{'source_sha256':'other'},{'design_signature':'other'}])
def test_resume_refuses_inconsistent_identity(change):
    import experiments as e
    from scripts.spec_baseline_common import validate_resume
    identity={'source_sha256':'source','data_manifest_sha256':'data'}
    state=dict(step=5,tokens=5*e.BASELINE_STEP_TOKENS,design_signature='sig',**identity)
    assert validate_resume(state,'sig',identity)==5
    state.update(change)
    with pytest.raises(RuntimeError,match='resume identity'):validate_resume(state,'sig',identity)


def test_completed_baseline_refuses_changed_checkpoint_provenance(tmp_path,monkeypatch):
    import json,experiments as e
    from scripts import spec_baseline_common as common
    exp=e.ExperimentConfig('head','medusa',horizons=4)
    path=tmp_path/'final.pt';marker=tmp_path/'trained.json'
    identity=dict(source_sha256='source',data_manifest_sha256='data')
    state=dict(recipe='medusa',step=e.BASELINE_STEPS,tokens=e.BASELINE_STEPS*e.BASELINE_STEP_TOKENS,
        design_signature='sig',model={},**identity)
    torch.save(state,path)
    marker.write_text(json.dumps(dict(checkpoint=str(path),design_signature='sig',**identity)))
    monkeypatch.setattr(e,'training_marker',lambda exp:marker)
    monkeypatch.setattr(e,'design_signature',lambda:'sig')
    monkeypatch.setattr(common,'validate_evaluation_inputs',lambda trained:None)
    common.checked_state(exp)
    state['source_sha256']='different';torch.save(state,path)
    with pytest.raises(RuntimeError,match='provenance'):common.checked_state(exp)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA inference dtype/cache')
def test_ar_cuda_inference_without_autocast():
    torch.manual_seed(6);model=TinyAR(tiny_config()).cuda().eval()
    ids=torch.tensor([[1,2,3,16,4]],device='cuda')
    with torch.inference_mode():
        full=model(ids);cache=None;parts=[]
        for i in range(ids.shape[1]):
            cache,logits=model.encode(ids[:,i:i+1],cache);parts.append(logits)
        torch.testing.assert_close(full,torch.cat(parts,1),atol=1e-3,rtol=1e-2)
        assert torch.isfinite(full).all()
