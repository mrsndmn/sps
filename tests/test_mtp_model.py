"""Tiny CPU oracles for the frozen group-local SPS extension."""
import types

import pytest
import torch
from torch.nn import functional as F

from modeling.models.model import apply_rotary_emb
from modeling.models.sps.core import SPSConfig, SPSModelBase
from modeling.models.sps.mtp import SPSMTP, truncate_cache


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny(n_head=2, hidden_size=16):
    torch.manual_seed(41)
    return SPSModelBase(SPSConfig(vocab_size=19, block_size=8, n_layer=2,
        n_head=n_head, hidden_size=hidden_size, intermediate_size=24, eos_token_id=16,
        pad_token_id=17, predict_token_id=18, window_size=0, dropout=0.0))


def dense_source_attention(self, x, freqs_cis, documents_idx_Bx2T=None):
    b, t, c = x.shape
    q, k, v = [a.view(b,t,self.n_head,c//self.n_head) for a in self.c_attn(x).chunk(3,-1)]
    q,k = apply_rotary_emb(q,k,freqs_cis)
    q,k,v = [a.transpose(1,2).to(torch.bfloat16) for a in (q,k,v)]
    i = torch.arange(t,device=x.device)
    allowed = (i[:,None]>=i[None,:]) & ((i[None,:]%2==0) | (i[:,None]==i[None,:]))
    allowed = allowed[None] & (documents_idx_Bx2T[:,:,None] == documents_idx_Bx2T[:,None,:])
    y=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed[:,None])
    return self.c_proj(y.transpose(1,2).reshape(b,t,c).to(x.dtype))


def test_p1_matches_original_dense_sps_and_n1():
    reference=tiny()
    for block in reference.transformer.h:
        block.attn.forward=types.MethodType(dense_source_attention,block.attn)
    ids=torch.tensor([[17,1,2,16,3,4],[17,17,1,2,3,4]])
    expected=reference(ids)
    for n in (1,2,4):
        model=SPSMTP(reference,n,backend='sdpa')
        actual=model(ids)
        assert actual.shape==(2,6,n,19)
        torch.testing.assert_close(actual[:,:,0],expected,atol=2e-5,rtol=2e-4)


def test_only_new_embeddings_update_with_every_horizon_gradient():
    model=SPSMTP(tiny(),4,backend='sdpa')
    assert [name for name,p in model.named_parameters() if p.requires_grad]==['new_embeddings']
    before={name:p.clone() for name,p in model.reference.named_parameters()}
    original=model.new_embeddings.detach().clone()
    optimizer=torch.optim.AdamW([model.new_embeddings],lr=.01,weight_decay=.1)
    logits=model(torch.tensor([[1,2,3,4]]))
    F.cross_entropy(logits[:,:,1:].reshape(-1,19),torch.arange(12)%16).backward()
    assert torch.isfinite(model.new_embeddings.grad).all()
    assert (model.new_embeddings.grad.abs().sum(-1)>0).all()
    optimizer.step()
    assert not torch.equal(original,model.new_embeddings)
    for name,p in model.reference.named_parameters():
        assert torch.equal(before[name],p),name
        assert p.grad is None


def test_future_and_document_isolation_and_selected_groups():
    model=SPSMTP(tiny(),3,backend='sdpa')
    ids=torch.tensor([[17,1,2,16,3,4]])
    baseline=model(ids)
    altered=ids.clone(); altered[:,4:]=torch.tensor([7,8])
    torch.testing.assert_close(model(altered)[:,:4],baseline[:,:4],atol=0,rtol=0)
    altered=ids.clone(); altered[:,1:3]=torch.tensor([7,8])
    torch.testing.assert_close(model(altered)[:,4:],baseline[:,4:],atol=0,rtol=0)
    positions=torch.tensor([1,4,5])
    torch.testing.assert_close(model.forward_selected(ids,positions),baseline[:,positions],atol=2e-4,rtol=.005)
    # Removing all other groups cannot affect this group's prediction states.
    torch.testing.assert_close(model.forward_selected(ids,positions[1:2]),baseline[:,4:5],atol=2e-4,rtol=.005)
    # Earlier padding and its fake group are invisible to the real suffix.
    short=model(ids[:,1:])
    torch.testing.assert_close(short,baseline[:,1:],atol=2e-4,rtol=.005)


def test_cache_incremental_and_rollback_with_eos():
    model=SPSMTP(tiny(),4,backend='sdpa')
    ids=torch.tensor([[17,1,2,16,3,4]])
    with torch.no_grad():
        full=model(ids)
        cache,a=model.encode(ids[:,:3])
        cache,b=model.encode(ids[:,3:],cache)
        torch.testing.assert_close(torch.cat((a,b),1),full,atol=2e-5,rtol=2e-4)
        assert cache.length==6
        recovered=truncate_cache(cache,2)
        assert recovered.length==2 and cache.length==6
        recovered,c=model.encode(ids[:,2:],recovered)
        torch.testing.assert_close(c,full[:,2:],atol=2e-5,rtol=2e-4)
        assert all(not k.requires_grad and not v.requires_grad for k,v in recovered.layers)
    with pytest.raises(ValueError):
        truncate_cache(cache,7)


def test_bounds_and_invalid_source():
    model=SPSMTP(tiny(),4)
    assert model(torch.tensor([[1]*8])).shape==(1,8,4,19)
    with pytest.raises(ValueError):
        model(torch.tensor([[1]*9]))
    with pytest.raises(ValueError):
        SPSMTP(tiny(),0)
    reference=tiny(); reference.config.window_size=1
    with pytest.raises(ValueError):
        SPSMTP(reference,2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_flex_forward_backward_and_original_p1():
    reference=tiny(n_head=1, hidden_size=64).cuda()
    model=SPSMTP(reference,3,backend='flex').cuda()
    ids=torch.tensor([[17,1,2,16,3,4]],device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        actual=model(ids)
        model.backend='sdpa'
        expected=model(ids)
        source=reference(ids)
        model.backend='flex'
        # Reduced precision tiled kernels can round their output differently.
        torch.testing.assert_close(actual,expected,atol=.003,rtol=.02)
        torch.testing.assert_close(actual[:,:,0],source,atol=.003,rtol=.02)
        F.cross_entropy(actual[:,:,1:].float().reshape(-1,19),
                        torch.arange(12,device='cuda')%16).backward()
    assert torch.isfinite(model.new_embeddings.grad).all()
    assert (model.new_embeddings.grad.abs().sum(-1)>0).all()
    model.backend='auto'
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        cache,a=model.encode(ids[:,:3])
        cache,b=model.encode(ids[:,3:],cache)
        torch.testing.assert_close(torch.cat((a,b),1),expected,atol=.003,rtol=.02)
        cache=truncate_cache(cache,2)
        _,recovered=model.encode(ids[:,2:],cache)
        torch.testing.assert_close(recovered,expected[:,2:],atol=.003,rtol=.02)


def test_actual_hidden_perturbation_cannot_cross_prediction_groups():
    model=SPSMTP(tiny(),3,backend='sdpa')
    ids=torch.tensor([[1,2,3,4]])
    baseline=model(ids)
    def perturb_other_groups(module,args):
        (x,)=args
        if x.shape[1]==12:  # prediction stream, after the real stream was frozen
            x=x.clone()
            x[:,:6]+=torch.arange(16,dtype=x.dtype)*10
            return (x,)
    handle=model.reference.transformer.h[0].attn.c_attn.register_forward_pre_hook(perturb_other_groups)
    altered=model(ids)
    handle.remove()
    torch.testing.assert_close(altered[:,2:],baseline[:,2:],atol=0,rtol=0)
    assert not torch.equal(altered[:,:2],baseline[:,:2])
    grad=torch.autograd.grad(baseline[:,:,0].sum(),model.new_embeddings)[0]
    assert torch.count_nonzero(grad)==0


def test_mask_permissions_for_all_groups_and_documents():
    model=SPSMTP(tiny(),4,backend='sdpa')
    ids=torch.tensor([[17,1,16,2]])
    docs=model.reference.generate_document_idx(ids)
    _,mask=model._mask(docs,torch.arange(4)[None],predict=True)
    mask=mask[0,0]
    for qi in range(16):
        group,horizon=divmod(qi,4)
        for ki in range(20):
            if ki<4:
                expected=ki<=group and docs[0,ki]==docs[0,group]
            else:
                kg,kh=divmod(ki-4,4)
                expected=kg==group and kh<=horizon
            assert bool(mask[qi,ki])==bool(expected),(qi,ki)


def test_batched_selected_positions_and_zero_length_cache():
    model=SPSMTP(tiny(),2,backend='sdpa')
    ids=torch.tensor([[1,2,3],[4,5,6]])
    positions=torch.tensor([[0,2],[1,2]])
    expected=model(ids)
    actual=model.forward_selected(ids,positions)
    torch.testing.assert_close(actual,expected[torch.arange(2)[:,None],positions],atol=2e-4,rtol=.005)
    with torch.no_grad():
        cache,_=model.encode(ids,positions=[2])
        empty=truncate_cache(cache,0)
        _,actual=model.encode(ids,empty,positions=[2])
        torch.testing.assert_close(actual,expected[:,2:3],atol=2e-4,rtol=.005)


def test_actual_padding_hidden_states_are_isolated():
    model=SPSMTP(tiny(),3,backend='sdpa')
    ids=torch.tensor([[17,17,1,2]])
    baseline=model(ids)
    def perturb_padding(module,args,output):
        if output.shape[1]==4:
            output=output.clone()
            output[:,:2]+=100
            return output
    handle=model.reference.transformer.h[0].attn.c_attn.register_forward_hook(perturb_padding)
    actual=model(ids)
    handle.remove()
    torch.testing.assert_close(actual[:,2:],baseline[:,2:],atol=0,rtol=0)


def test_unfrozen_loss_reaches_real_embeddings_and_real_only_mlp_states():
    model = SPSMTP(tiny(), 3, backend='sdpa', train_backbone=True)
    real_states = {}

    def capture_embedding(module, args, output):
        real_states['embedding'] = output

    def capture_real_mlp(module, args, output):
        if output.shape[1] == 4:
            real_states['mlp'] = output

    handles = [model.reference.transformer.wte.register_forward_hook(capture_embedding),
               model.reference.transformer.h[0].mlp.register_forward_hook(capture_real_mlp)]
    try:
        logits = model(torch.tensor([[1, 2, 3, 4]]))
    finally:
        for handle in handles:
            handle.remove()
    loss = F.cross_entropy(logits[:, -1].reshape(-1, 19), torch.tensor([5, 6, 7]))
    assert all(state.requires_grad for state in real_states.values())
    grads = torch.autograd.grad(loss, tuple(real_states.values()))
    assert all(torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in grads)


def test_unfrozen_appended_loss_reaches_past_real_keys_and_values():
    model = SPSMTP(tiny(), 3, backend='sdpa', train_backbone=True)
    cache, _ = model.encode(torch.tensor([[1, 2]]))
    past = tuple(state for layer in cache.layers for state in layer)
    assert all(state.requires_grad for state in past)
    _, logits = model.encode(torch.tensor([[3, 4]]), cache, positions=[3])
    loss = F.cross_entropy(logits.reshape(-1, 19), torch.tensor([5, 6, 7]))
    grads = torch.autograd.grad(loss, past)
    assert all(torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in grads)


def test_unfrozen_tied_embedding_updates_once_and_modes_control_real_graph():
    model = SPSMTP(tiny(), 3, backend='sdpa', train_backbone=True)
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert model.reference.training
    weight = model.reference.transformer.wte.weight
    assert model.reference.lm_head.weight is weight
    parameters = list(model.parameters())
    assert sum(parameter is weight for parameter in parameters) == 1
    before = weight.detach().clone()
    optimizer = torch.optim.SGD(parameters, lr=.1)
    logits = model(torch.tensor([[1, 2, 3, 4]]))
    F.cross_entropy(logits.reshape(-1, 19), torch.arange(12) % 16).backward()
    expected = before - .1 * weight.grad
    assert model.new_embeddings.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in model.reference.parameters())
    optimizer.step()
    torch.testing.assert_close(weight, expected)
    assert not torch.equal(weight, before)
    model.eval()
    assert not model.reference.training
    cache, _ = model.encode(torch.tensor([[1, 2]]))
    assert all(not state.requires_grad for layer in cache.layers for state in layer)
    model.train()
    assert model.reference.training
    with torch.no_grad():
        cache, _ = model.encode(torch.tensor([[1, 2]]))
    assert all(not state.requires_grad for layer in cache.layers for state in layer)
    frozen = SPSMTP(tiny(), 3, backend='sdpa').train()
    assert not frozen.reference.training


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_unfrozen_flex_backward_through_real_and_prediction_kv():
    model = SPSMTP(tiny(n_head=1, hidden_size=64).cuda(), 3,
                   backend='flex', train_backbone=True).cuda()
    ids = torch.tensor([[1, 2, 3, 4]], device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        cache, _ = model.encode(ids[:, :2])
        past = tuple(state for layer in cache.layers for state in layer)
        for state in past:
            state.retain_grad()
        _, logits = model.encode(ids[:, 2:], cache, positions=[3])
        loss = F.cross_entropy(logits.float().reshape(-1, 19),
                               torch.tensor([5, 6, 7], device='cuda'))
    loss.backward()
    assert all(state.grad is not None and torch.isfinite(state.grad).all()
               and state.grad.abs().sum() > 0 for state in past)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
