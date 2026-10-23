"""Layer-parallel, batch-one verification for zero-window SPS.

Every draft token has its own real/predict pair. Attention uses the native q2
kernel with the token index as the batch dimension and causal per-row KV lengths.
Only real KV is retained. Rollback slices the verified branch, never replays an
accepted draft token. Numerical agreement with native decoding is measured by the
benchmark: different GEMM shapes need not give bitwise identical floating outputs.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F
from modeling.models.model import apply_rotary_emb
from modeling.models.sps.mtp import MTPCache, truncate_cache
from modeling.models.utils.decode_attention import KVSegment, triton_segmented_q2_attention


@torch.inference_mode()
def block_decode(reference, cache, tokens):
    """Append [T] real tokens in one layer traversal; return cache and [T,V]."""
    if tokens.ndim != 1 or tokens.numel() == 0:
        raise ValueError('Expected nonempty one-dimensional block')
    cfg = reference.config
    if cfg.window_size != 0 or cfg.dropout != 0 or any(
            b.attn.persistent_key_window is not None for b in reference.transformer.h):
        raise ValueError('Parallel verification requires unrestricted zero-window SPS')
    old = cache.length if cache is not None else 0
    ids = tokens[None] if cache is None else torch.cat((cache.input_ids, tokens[None]), 1)
    if ids.shape[1] > cfg.block_size or bool((ids == cfg.pad_token_id).any()):
        raise ValueError('Block exceeds context or contains padding')
    n, h, d = tokens.numel(), cfg.n_head, cfg.hidden_size // cfg.n_head
    positions = torch.arange(old, old+n, device=tokens.device)
    # EOS belongs to its preceding document; only the following token resets.
    indices = torch.arange(ids.shape[1], device=tokens.device)
    starts = torch.cat((torch.zeros(1,device=tokens.device,dtype=torch.long),
                        torch.where(ids[0,:-1] == cfg.eos_token_id, indices[:-1]+1, 0))).cummax(0).values
    start = starts[old:]
    lengths = positions-start
    # Pack each row's preceding document keys at zero, matching native reduction
    # boundaries. Include current slots separately, exactly as q2 decode does.
    scan = ids.shape[1]
    gather = (start[:,None]+torch.arange(scan,device=tokens.device)[None]).clamp(max=scan-1)
    pair = torch.stack((tokens, torch.full_like(tokens,cfg.predict_token_id)), 1)
    x = reference.transformer.wte(pair)
    freqs = reference.freqs_cis[positions][:,None].expand(-1,2,-1)
    layers = []
    for i, block in enumerate(reference.transformer.h):
        q,k,v = [t.reshape(n,2,h,d) for t in block.attn.c_attn(block.attention_norm(x)).chunk(3,-1)]
        q,k = apply_rotary_emb(q,k,freqs)
        dtype = torch.bfloat16
        q,k,v = (t.to(dtype) for t in (q,k,v))
        real_k = k[:,0].transpose(0,1)[None]
        real_v = v[:,0].transpose(0,1)[None]
        if cache is not None:
            real_k = torch.cat((cache.layers[i][0],real_k),2)
            real_v = torch.cat((cache.layers[i][1],real_v),2)
        packed_k = real_k.expand(n,-1,-1,-1).gather(2,gather[:,None,:,None].expand(n,h,scan,d)).contiguous()
        packed_v = real_v.expand(n,-1,-1,-1).gather(2,gather[:,None,:,None].expand(n,h,scan,d)).contiguous()
        if tokens.is_cuda:
            attn = triton_segmented_q2_attention(q,[KVSegment(packed_k,packed_v,lengths,scan)],
                current_normal_k_BxHxD=k[:,0],current_normal_v_BxHxD=v[:,0],
                current_predict_k_BxHxD=k[:,1],current_predict_v_BxHxD=v[:,1],attn_dtype=dtype)
        else:
            keys = torch.cat((packed_k,k.transpose(1,2)),2)
            values = torch.cat((packed_v,v.transpose(1,2)),2)
            ki = torch.arange(scan+2,device=tokens.device)
            mask = (ki[None,None,:] < lengths[:,None,None]).expand(n,2,-1).clone()
            mask[:,:,scan] = True
            mask[:,1,scan+1] = True
            attn = F.scaled_dot_product_attention(q.transpose(1,2),keys,values,
                attn_mask=mask[:,None]).transpose(1,2)
        y = attn.reshape(n,2,cfg.hidden_size).to(block.attn.c_proj.weight.dtype)
        x = x+block.attn.c_proj(y)
        x = x+block.mlp(block.mlp_norm(x))
        layers.append((real_k,real_v))
    logits = reference.lm_head(reference.transformer.output_norm(x)[:,1])
    return MTPCache(ids,tuple(layers)),logits


class ParallelPrefixVerifier:
    """A speculative branch retaining every verified real-prefix KV entry."""
    backend = 'parallel_q2'

    def __init__(self, reference, max_new_tokens=128):
        self.reference = reference
        self.max_new_tokens = max_new_tokens
        self.cache = None
        self.prefix = None
        self.logits = None
        self.block_calls = 0

    def _prefill(self, prefix):
        # Use the original prefill kernels, preserving the common initial state.
        if prefix.is_cuda:
            budget = min(self.max_new_tokens,self.reference.config.block_size-prefix.numel())
            state,logits,_ = self.reference._prefill_generation_state(prefix[None],budget,require_batched=True)
            start = 0
            eos = (prefix[:-1] == self.reference.config.eos_token_id).nonzero().flatten()
            if eos.numel(): start = int(eos[-1])+1
            layers=[]
            for k,v in zip(state.predict_retained.k,state.predict_retained.v):
                pk=k.new_zeros((1,k.shape[1],prefix.numel(),k.shape[-1]));pv=torch.zeros_like(pk)
                pk[:,:,start:] = k[:,:,:prefix.numel()-start]
                pv[:,:,start:] = v[:,:,:prefix.numel()-start]
                layers.append((pk,pv))
            self.cache=MTPCache(prefix[None].clone(),tuple(layers))
            self.logits=logits
        else:
            self.cache,self.logits=block_decode(self.reference,None,prefix)
        self.prefix=prefix.clone()

    @torch.inference_mode()
    def ensure(self,prefix):
        if self.cache is None:
            self._prefill(prefix)
            return self.logits[-1]
        offset=self.logit_offset
        limit=min(prefix.numel(),self.prefix.numel())
        wrong=(prefix[:limit] != self.prefix[:limit]).nonzero().flatten()
        common=int(wrong[0]) if wrong.numel() else limit
        if common <= offset:
            self._prefill(prefix)
        elif common == prefix.numel():
            # Cached logits align with every prefix in the speculative branch.
            self.cache=truncate_cache(self.cache,common)
            self.prefix=self.prefix[:common]
            self.logits=self.logits[:common-offset]
        else:
            self.cache=truncate_cache(self.cache,common)
            self.cache,new=block_decode(self.reference,self.cache,prefix[common:])
            self.block_calls+=1
            retained=max(0,common-offset)
            self.logits=torch.cat((self.logits[:retained],new),0)
            self.prefix=prefix.clone()
        return self.logits[-1]

    @property
    def logit_offset(self):
        return self.prefix.numel()-self.logits.shape[0]

    @torch.inference_mode()
    def verify_logits(self,prefix,draft):
        # Fuse the previous round's correction/bonus with this draft. Its
        # logits are p[0]; running ensure first would add a serial target pass.
        if self.cache is not None:
            limit=min(prefix.numel(),self.prefix.numel())
            wrong=(prefix[:limit] != self.prefix[:limit]).nonzero().flatten()
            common=int(wrong[0]) if wrong.numel() else limit
            offset=self.logit_offset
            if offset < common < prefix.numel():
                pending=prefix.numel()-common
                self.cache=truncate_cache(self.cache,common)
                self.cache,new=block_decode(self.reference,self.cache,
                    torch.cat((prefix[common:],draft)))
                self.block_calls+=1
                self.logits=torch.cat((self.logits[:common-offset],new),0)
                self.prefix=torch.cat((prefix,draft))
                return new[pending-1:]
        first=self.ensure(prefix)
        self.cache,new=block_decode(self.reference,self.cache,draft)
        self.block_calls+=1
        self.logits=torch.cat((self.logits,new),0)
        self.prefix=torch.cat((prefix,draft))
        return torch.cat((first[None],new),0)
