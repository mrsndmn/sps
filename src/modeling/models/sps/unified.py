"""One-backbone SPS-MTP: fused real/group states and self-verification.

P1 is the target; P2..PN propose after a pending target token. Each verification
block computes all horizon groups in one layer traversal, retaining real-only KV
and horizon logits for every possible accepted prefix. No separate draft model.
"""
import torch
from torch.nn import functional as F
from .mtp import MTPCache,truncate_cache


@torch.inference_mode()
def unified_block(model,cache,tokens):
    if tokens.ndim!=1 or tokens.numel()==0:raise ValueError('Expected a nonempty token block')
    cfg=model.config;n=model.horizons;r=tokens.numel();old=cache.length
    ids=torch.cat((cache.input_ids,tokens[None]),1)
    if ids.shape[1]>cfg.block_size or bool((tokens==cfg.pad_token_id).any()):
        raise ValueError('Context overflow or padding in generation')
    docs=model.reference.generate_document_idx(ids)
    pos=torch.arange(old,old+r,device=tokens.device)
    group=torch.arange(r,device=tokens.device).repeat_interleave(n+1)
    slot=torch.arange(n+1,device=tokens.device).repeat(r)
    absolute=pos[group]
    rope=absolute+(slot-1).clamp_min(0)
    embeddings=torch.cat((model.reference.transformer.wte.weight[cfg.predict_token_id][None],model.new_embeddings),0)
    real=model.reference.transformer.wte(tokens)
    x=torch.cat((real[:,None],embeddings[None].expand(r,-1,-1)),1).reshape(1,r*(n+1),-1)
    qdoc=docs[:,absolute]
    historical=(qdoc[:,:,None]==docs[:,:old,None].transpose(1,2))
    # Real queries see causal real keys. Prediction queries also see preceding
    # predict slots within their own group, including themselves.
    causal_real=(slot[None,:]==0)&(absolute[None,:]<=absolute[:,None])
    local_predict=(slot[:,None]>0)&(slot[None,:]>0)&(group[:,None]==group[None,:])&(slot[None,:]<=slot[:,None])
    current=(causal_real|local_predict)[None]&(qdoc[:,:,None]==qdoc[:,None,:])
    mask=torch.cat((historical,current),-1)[:,None]
    layers=[]
    for block,(old_k,old_v) in zip(model.reference.transformer.h,cache.layers):
        q,k,v=model._qkv(block,x,rope[None])
        attended=F.scaled_dot_product_attention(q,torch.cat((old_k,k),2),torch.cat((old_v,v),2),attn_mask=mask)
        x=model._residual(block,x,attended)
        layers.append((torch.cat((old_k,k[:,:,::n+1]),2),torch.cat((old_v,v[:,:,::n+1]),2)))
    hidden=model.reference.transformer.output_norm(x.reshape(1,r,n+1,-1)[:,:,1:])
    return MTPCache(ids,tuple(layers)),model.reference.lm_head(hidden)[0]


class UnifiedSPS:
    """A single model/cache, with the final correction/bonus kept pending."""
    def __init__(self,model):
        if model.horizons<2:raise ValueError('Self speculation requires at least two horizons')
        self.model=model;self.cache=None;self.logits=None;self.offset=0;self.block_calls=0;self.prefills=0

    @torch.inference_mode()
    def ensure(self,prefix):
        if prefix.ndim!=1 or prefix.numel()==0:raise ValueError('Expected a nonempty prefix')
        common=0
        if self.cache is not None:
            limit=min(prefix.numel(),self.cache.length)
            wrong=(self.cache.input_ids[0,:limit]!=prefix[:limit]).nonzero().flatten()
            common=int(wrong[0]) if wrong.numel() else limit
        if self.cache is None or common<=self.offset:
            self.cache,logits=self.model.encode(prefix[None],positions=[prefix.numel()-1])
            self.logits=logits[0];self.offset=prefix.numel()-1;self.prefills+=1
        else:
            self.cache=truncate_cache(self.cache,common)
            self.logits=self.logits[:common-self.offset]
            if common<prefix.numel():
                self.cache,new=unified_block(self.model,self.cache,prefix[common:]);self.block_calls+=1
                self.logits=torch.cat((self.logits,new),0)
        return self.logits[-1]

    def propose(self,history):
        # The last token is an already emitted anchor/correction/bonus. Its
        # predecessor's group is available from the preceding verification.
        return self.ensure(history[:-1])[1:]

    @torch.inference_mode()
    def verify_logits(self,history,draft):
        self.ensure(history[:-1])
        self.cache,new=unified_block(self.model,self.cache,torch.cat((history[-1:],draft)))
        self.block_calls+=1;self.logits=torch.cat((self.logits,new),0)
        return new[:,0]
