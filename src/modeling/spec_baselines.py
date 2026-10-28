"""Lightweight proposal models; data and experiment settings live in experiments.py."""
from collections import Counter
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from modeling.models.model import RMSNorm, MLP, precompute_freqs_cis, apply_rotary_emb
from modeling.models.sps.mtp import MTPCache


class MedusaHeads(nn.Module):
    """Independent residual heads for future offsets 2..K+1, sharing frozen LM head."""
    def __init__(self,width,horizons):
        super().__init__()
        self.heads=nn.ModuleList([nn.Linear(width,width,bias=False) for _ in range(horizons)])
        for head in self.heads:nn.init.zeros_(head.weight)

    def forward(self,features,lm_head):
        hidden=torch.stack([features+F.silu(head(features)) for head in self.heads],-2)
        return lm_head(hidden)


class TinyAR(nn.Module):
    """Causal RoPE Transformer with document-safe SDPA and reusable real-token KV."""
    def __init__(self,config):
        super().__init__();self.config=config;self.horizons=1
        self.wte=nn.Embedding(config.vocab_size,config.hidden_size)
        self.layers=nn.ModuleList([nn.ModuleDict(dict(norm=RMSNorm(config),
            qkv=nn.Linear(config.hidden_size,3*config.hidden_size,bias=False),
            out=nn.Linear(config.hidden_size,config.hidden_size,bias=False),
            mlp_norm=RMSNorm(config),mlp=MLP(config))) for _ in range(config.n_layer)])
        self.norm=RMSNorm(config)
        self.lm_head=nn.Linear(config.hidden_size,config.vocab_size,bias=False)
        self.lm_head.weight=self.wte.weight
        self.register_buffer('freqs',precompute_freqs_cis(config.hidden_size//config.n_head,
            config.block_size),persistent=False)
        for p in self.parameters():
            if p.ndim>=2:nn.init.normal_(p,std=.02)

    def encode(self,tokens,cache=None,*,positions=None):
        old=cache.length if cache is not None else 0
        ids=tokens if cache is None else torch.cat((cache.input_ids,tokens),1)
        b,n=tokens.shape;total=ids.shape[1]
        if total>self.config.block_size or n==0:raise ValueError('Invalid AR context length')
        docs=F.pad((ids[:,:-1]==self.config.eos_token_id).long().cumsum(1),(1,0))
        qi=torch.arange(old,total,device=tokens.device)
        ki=torch.arange(total,device=tokens.device)
        mask=((ki[None,None,:]<=qi[None,:,None]) &
            (docs[:,old:,None]==docs[:,None,:]) &
            (ids[:,None,:]!=self.config.pad_token_id))[:,None]
        x=self.wte(tokens);layers=[];h=self.config.n_head;d=self.config.hidden_size//h
        for i,layer in enumerate(self.layers):
            q,k,v=[a.reshape(b,n,h,d) for a in layer['qkv'](layer['norm'](x)).chunk(3,-1)]
            q,k=apply_rotary_emb(q,k,self.freqs[qi])
            q,k,v=(a.transpose(1,2) for a in (q,k,v))
            if tokens.is_cuda:q,k,v=(a.to(torch.bfloat16) for a in (q,k,v))
            if cache is not None:
                k=torch.cat((cache.layers[i][0],k),2);v=torch.cat((cache.layers[i][1],v),2)
            a=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
            x=x+layer['out'](a.transpose(1,2).reshape(b,n,-1).to(layer['out'].weight.dtype))
            x=x+layer['mlp'](layer['mlp_norm'](x));layers.append((k,v))
        if positions is not None:
            index=torch.as_tensor(positions,device=tokens.device)-old
            if bool(((index<0)|(index>=n)).any()):raise ValueError('AR positions outside appended block')
            x=x[:,index]
        logits=self.lm_head(self.norm(x))[:,:,None]
        return MTPCache(ids,tuple(layers)),logits

    def forward(self,tokens):return self.encode(tokens)[1]


def lookup_proposal(prefix,horizons,vocabulary,*,max_match):
    """Copy a continuation found entirely in history; deterministic unigram fallback."""
    history=prefix.tolist();draft=[]
    for size in range(min(max_match,len(history)-1),0,-1):
        suffix=history[-size:]
        for start in range(len(history)-size-1,-1,-1):
            if history[start:start+size]==suffix:
                draft=history[start+size:start+size+horizons]
                break
        if draft:break
    fallback=Counter(history).most_common(1)[0][0]
    draft=(draft+[fallback]*horizons)[:horizons]
    tokens=torch.tensor(draft,device=prefix.device,dtype=torch.long)
    q=torch.zeros(horizons,vocabulary,device=prefix.device)
    q.scatter_(1,tokens[:,None],1.)
    return tokens,q
