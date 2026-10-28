import pytest
import torch
from modeling.models.sps.parallel_verify import ParallelPrefixVerifier
from test_mtp_evaluate import model


def test_block_causality_and_rejection_cpu():
    ref = model().reference
    verifier = ParallelPrefixVerifier(ref)
    prefix = torch.tensor([1, 2, 3])
    draft = torch.tensor([5, 16, 7])
    raw = verifier.verify_logits(prefix, draft)
    for j in range(4):
        ids = torch.cat((prefix, draft[:j]))
        state, expected, _ = ref._prefill_generation_state(ids[None], 4)
        torch.testing.assert_close(raw[j], expected[0], atol=2e-5, rtol=2e-4)
    corrected = torch.tensor([1, 2, 3, 5, 9])
    actual = verifier.ensure(corrected)
    _, expected, _ = ref._prefill_generation_state(corrected[None], 4)
    torch.testing.assert_close(actual, expected[0], atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize('draft',[[5],[5,6],[5,6,7,8],[16,3,4],[5,16,7]])
def test_future_mask_and_rollback_to_each_accepted_length(draft):
    from modeling.models.sps.parallel_verify import block_decode
    ref=model().reference
    p=torch.tensor([1,2,3]);d=torch.tensor(draft)
    v=ParallelPrefixVerifier(ref)
    raw=v.verify_logits(p,d)
    altered=d.clone();altered[-1]=(altered[-1]+1)%16
    other=ParallelPrefixVerifier(ref).verify_logits(p,altered)
    torch.testing.assert_close(raw[:-1],other[:-1],atol=0,rtol=0)
    for accepted in range(len(d)+1):
        v=ParallelPrefixVerifier(ref);v.verify_logits(p,d)
        correction=torch.tensor([9])
        current=torch.cat((p,d[:accepted],correction))
        got=v.ensure(current)
        _,want=block_decode(ref,None,current)
        torch.testing.assert_close(got,want[-1],atol=3e-5,rtol=3e-4)
        assert torch.equal(v.cache.input_ids[0],current)
        # Repeated reconciliation must not re-run verified work.
        calls=v.block_calls
        torch.testing.assert_close(v.ensure(current),got,atol=0,rtol=0)
        assert v.block_calls==calls


def test_parallel_short_budgets_and_greedy_oracle():
    from scripts.mtp_evaluate import SpeculativeAdapter, probabilities
    from modeling.mtp_speculative import speculative_generate
    mtp=model();prefix=torch.tensor([1,2])
    for budget in (0,1,2,3,7):
        adapter=SpeculativeAdapter(mtp,verifier_factory=ParallelPrefixVerifier)
        actual=speculative_generate(prefix,adapter.propose,adapter.verify,
            max_new_tokens=budget,eos_token_id=16)
        expected=[]
        with torch.inference_mode():
            for _ in range(budget):
                ids=torch.cat((prefix,torch.tensor(expected,dtype=torch.long)))
                _,logits,_=mtp.reference._prefill_generation_state(ids[None],1)
                token=probabilities(logits,mtp.config).argmax(-1).item();expected.append(token)
                if token==16:break
        assert actual.token_ids.tolist()==expected


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA native oracle')
def test_cuda_block_native_logits_and_eos_recovery():
    from modeling.models.sps import SPSConfig,SPSModel
    from scripts.mtp_evaluate import NativePrefixVerifier,probabilities
    torch.manual_seed(23)
    ref=SPSModel(SPSConfig(vocab_size=39,block_size=32,n_layer=2,n_head=2,
        hidden_size=128,intermediate_size=192,eos_token_id=36,pad_token_id=37,
        predict_token_id=38,window_size=0,dropout=0.)).cuda().eval()
    prefix=torch.tensor([1,36,3,4],device='cuda')
    draft=torch.tensor([5,36,7,8],device='cuda')
    v=ParallelPrefixVerifier(ref,max_new_tokens=12)
    native=NativePrefixVerifier(ref,max_new_tokens=12)
    actual=v.verify_logits(prefix,draft);expected=native.verify_logits(prefix,draft)
    torch.testing.assert_close(probabilities(actual,ref.config),probabilities(expected,ref.config),atol=2e-4,rtol=2e-3)
    for ids in [torch.cat((prefix,draft[:1],draft.new_tensor([9]))),
                torch.cat((prefix,draft[:3],draft.new_tensor([10]))),prefix,
                torch.tensor([1,36],device='cuda')]:
        torch.testing.assert_close(probabilities(v.ensure(ids)[None],ref.config),
            probabilities(native.ensure(ids)[None],ref.config),atol=2e-4,rtol=2e-3)


def test_parallel_evaluation_registry_is_isolated():
    import subprocess,sys,json,os
    env=dict(os.environ,SPS_EXPERIMENT_SUITE='mtp-parallel')
    code='''import experiments as e,json
from scripts.mlspace_launch import payload
print(json.dumps(dict(root=str(e.ARTIFACTS_ROOT),models=[x.horizons for x in e.experiment_configs],parents=[str(e.training_marker(x)) for x in e.experiment_configs],payload=payload('eval',e.experiment_configs[0].name,e.STAGING_ROOT/'test','a','b'))))'''
    a=json.loads(subprocess.check_output([sys.executable,'-c',code],env=env,text=True))
    assert a['root'].endswith('/mtp-parallel') and a['models']==[2,3,4]
    assert all('/mtp-unfreeze/' in x for x in a['parents'])
    assert 'scripts.mtp_parallel_evaluate eval' in a['payload']['script']
    assert a['payload']['instance_type']=='a100.1gpu'
    r=subprocess.run([sys.executable,'run_train_jobs.py','--dry'],env=env,capture_output=True,text=True)
    assert r.returncode and 'evaluation-only' in r.stderr


def test_one_layer_traversal_per_block_and_no_replay_of_accepted_tokens():
    ref=model().reference;v=ParallelPrefixVerifier(ref)
    p=torch.tensor([1,2,3]);v.ensure(p)
    seen=[]
    handle=ref.transformer.h[0].attn.c_attn.register_forward_hook(lambda m,a,o:seen.append(a[0].shape[0]))
    try:
        d=torch.tensor([4,5,6]);v.verify_logits(p,d)
        assert seen==[3]
        v.ensure(torch.cat((p,d)))
        assert seen==[3]
        v.ensure(torch.cat((p,d[:2],torch.tensor([9]))))
        assert seen==[3,1]
    finally:handle.remove()


def test_correction_and_next_draft_share_one_target_pass():
    from modeling.models.sps.parallel_verify import block_decode
    ref=model().reference;v=ParallelPrefixVerifier(ref)
    p=torch.tensor([1,2,3]);d=torch.tensor([4,5,6])
    v.verify_logits(p,d)
    seen=[]
    handle=ref.transformer.h[0].attn.c_attn.register_forward_hook(lambda m,a,o:seen.append(a[0].shape[0]))
    corrected=torch.tensor([1,2,3,4,9])
    try:raw=v.verify_logits(corrected,d)
    finally:handle.remove()
    assert seen==[4]
    for j in range(4):
        _,want=block_decode(ref,None,torch.cat((corrected,d[:j])))
        torch.testing.assert_close(raw[j],want[-1],atol=3e-5,rtol=3e-4)
