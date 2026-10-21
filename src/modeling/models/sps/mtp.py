"""Frozen SPS with input-only, group-local multi-token prediction embeddings.

Real-token KV states never depend on a prediction slot. A prediction query has one
joint softmax over its causal real prefix and its own group's causal slots. The
CUDA backend uses compiled tiled FlexAttention, without a quadratic dense mask;
SDPA is the small-sequence CPU oracle. Cached tensors contain only real states.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from modeling.models.model import apply_rotary_emb, precompute_freqs_cis


@dataclass(frozen=True)
class MTPCache:
    input_ids: Tensor
    layers: tuple[tuple[Tensor, Tensor], ...]

    @property
    def length(self) -> int:
        return self.input_ids.shape[1]


def truncate_cache(cache: MTPCache, length: int) -> MTPCache:
    """Return a nonmutating real-prefix rollback; zero length is supported."""
    if not 0 <= length <= cache.length:
        raise ValueError(f"Invalid cache length {length} for prefix {cache.length}")
    return MTPCache(cache.input_ids[:, :length],
                    tuple((k[:, :, :length], v[:, :, :length]) for k, v in cache.layers))


# Compiling the kernel (rather than the model) preserves the eager frozen/grad
# boundary. FlexAttention's backward computes gradients for prediction Q/K/V.
_compiled_flex = None


class SPSMTP(nn.Module):
    """Extend a zero-window SPS reference without changing its vocabulary.

    ``positions`` in forward_selected/encode may be [S] or [B,S], and are absolute
    real-token indices. encode's default returns every newly appended prefix;
    supplying positions avoids projecting all prefixes to the large vocabulary.
    Cache reuse is for inference: reference tensors are immutable after creation.
    """

    def __init__(self, reference: nn.Module, horizons: int, backend: str = "auto"):
        super().__init__()
        if not isinstance(horizons, int) or horizons < 1:
            raise ValueError("horizons must be a positive integer")
        if backend not in {"auto", "sdpa", "flex"}:
            raise ValueError(f"Unknown attention backend: {backend}")
        if reference.config.window_size != 0 or reference.config.dropout != 0:
            raise ValueError("MTP requires a zero-window, zero-dropout SPS reference")
        if any(block.attn.persistent_key_window is not None
               for block in reference.transformer.h):
            raise ValueError("MTP requires unrestricted causal real-token attention")
        self.reference = reference
        self.config = reference.config
        self.horizons = horizons
        self.backend = backend
        reference.requires_grad_(False)
        reference.eval()
        initial = reference.transformer.wte.weight[self.config.predict_token_id].detach()
        self.new_embeddings = nn.Parameter(initial.repeat(horizons - 1, 1))
        self.register_buffer("freqs_cis", precompute_freqs_cis(
            self.config.hidden_size // self.config.n_head,
            self.config.block_size + horizons - 1).to(initial.device), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.reference.eval()
        return self

    def truncate_cache(self, cache: MTPCache, length: int) -> MTPCache:
        return truncate_cache(cache, length)

    def _qkv(self, block, x: Tensor, positions: Tensor):
        b, t, c = x.shape
        q, k, v = [a.reshape(b, t, self.config.n_head, c // self.config.n_head)
                   for a in block.attn.c_attn(block.attention_norm(x)).chunk(3, -1)]
        q, k = apply_rotary_emb(q, k, self.freqs_cis[positions])
        # Original SPS explicitly rounds its attention inputs to bfloat16.
        return tuple(a.transpose(1, 2).to(torch.bfloat16) for a in (q, k, v))

    def _mask(self, docs: Tensor, query_positions: Tensor, *, predict: bool):
        """Build once per stream and reuse across every transformer layer."""
        b, t = docs.shape
        s = query_positions.shape[1]
        n = self.horizons
        query_docs = docs.gather(1, query_positions)
        if predict:
            query_positions = query_positions.repeat_interleave(n, dim=1)
            query_docs = query_docs.repeat_interleave(n, dim=1)
        qlen = query_positions.shape[1]
        klen = t + s * n if predict else t

        def allowed(batch, head, qi, ki):
            del head
            # Block-mask construction probes padded tile coordinates too.
            qsafe = qi.clamp(max=qlen - 1)
            ksafe = ki.clamp(max=t - 1)
            real = ((ki < t) & (ki <= query_positions[batch, qsafe])
                    & (docs[batch, ksafe] == query_docs[batch, qsafe]))
            if predict:
                pi = ki - t
                local = ((ki >= t) & (qi // n == pi // n) & (qi % n >= pi % n))
                return real | local
            return real

        # Small cached decode queries use SDPA: its mask is only Q x K and
        # avoids compiling a different FlexAttention graph for each prefix
        # length. Long training/prefill queries use the tiled sparse backend.
        use_flex = self.backend == "flex" or (
            self.backend == "auto" and docs.is_cuda and qlen >= 128)
        if use_flex:
            if not docs.is_cuda:
                raise ValueError("FlexAttention backend requires CUDA")
            from torch.nn.attention.flex_attention import create_block_mask
            return "flex", create_block_mask(allowed, b, None, qlen, klen,
                                               device=docs.device, _compile=True)
        qi = torch.arange(qlen, device=docs.device)[None, :, None]
        ki = torch.arange(klen, device=docs.device)[None, None, :]
        batch = torch.arange(b, device=docs.device)[:, None, None]
        return "sdpa", allowed(batch, 0, qi, ki)[:, None]

    @staticmethod
    def _attention(q, k, v, mask):
        backend, actual_mask = mask
        if backend == "sdpa":
            return F.scaled_dot_product_attention(q, k, v, attn_mask=actual_mask)
        global _compiled_flex
        if _compiled_flex is None:
            from torch.nn.attention.flex_attention import flex_attention
            _compiled_flex = torch.compile(flex_attention, dynamic=False)
        return _compiled_flex(q, k, v, block_mask=actual_mask)

    @staticmethod
    def _residual(block, x: Tensor, attention: Tensor):
        b, t, c = x.shape
        y = attention.transpose(1, 2).reshape(b, t, c)
        y = block.attn.c_proj(y.to(block.attn.c_proj.weight.dtype))
        x = x + y
        return x + block.mlp(block.mlp_norm(x))

    def _positions(self, positions, batch: int, length: int, device):
        positions = torch.as_tensor(positions, device=device, dtype=torch.long)
        if positions.ndim == 1:
            positions = positions[None].expand(batch, -1)
        if positions.ndim != 2 or positions.shape[0] != batch or positions.shape[1] == 0:
            raise ValueError("positions must have shape [S] or [B,S], with S > 0")
        if bool(((positions < 0) | (positions >= length)).any()):
            raise ValueError("Prediction positions fall outside the real prefix")
        return positions

    def _encode_real(self, input_ids: Tensor, cache: MTPCache | None):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [B,T], with T > 0")
        previous = 0 if cache is None else cache.length
        if cache is not None and (cache.input_ids.shape[0] != input_ids.shape[0]
                                  or cache.input_ids.device != input_ids.device):
            raise ValueError("Cache batch/device mismatch")
        all_ids = input_ids if cache is None else torch.cat((cache.input_ids, input_ids), dim=1)
        b, total = all_ids.shape
        if total > self.config.block_size:
            raise ValueError(f"Real context {total} exceeds block size {self.config.block_size}")
        docs = self.reference.generate_document_idx(all_ids)
        positions = torch.arange(previous, total, device=input_ids.device)[None].expand(b, -1)
        real_mask = self._mask(docs, positions, predict=False)
        layers = []
        with torch.no_grad():
            x = self.reference.transformer.wte(input_ids)
            for i, block in enumerate(self.reference.transformer.h):
                q, k, v = self._qkv(block, x, positions)
                if cache is not None:
                    old_k, old_v = cache.layers[i]
                    k = torch.cat((old_k, k), dim=2)
                    v = torch.cat((old_v, v), dim=2)
                x = self._residual(block, x, self._attention(q, k, v, real_mask))
                layers.append((k.detach(), v.detach()))
        return MTPCache(all_ids.detach(), tuple(layers)), docs

    def _predict(self, cache: MTPCache, docs: Tensor, positions: Tensor):
        b, s = positions.shape
        n = self.horizons
        original = self.reference.transformer.wte.weight[self.config.predict_token_id][None]
        embeddings = torch.cat((original, self.new_embeddings), dim=0)
        x = embeddings[None, None].expand(b, s, n, -1).reshape(b, s * n, -1)
        rope_positions = (positions[:, :, None] + torch.arange(n, device=positions.device)).reshape(b, s * n)
        predict_mask = self._mask(docs, positions, predict=True)
        for block, (real_k, real_v) in zip(self.reference.transformer.h, cache.layers):
            q, k, v = self._qkv(block, x, rope_positions)
            k = torch.cat((real_k, k), dim=2)
            v = torch.cat((real_v, v), dim=2)
            x = self._residual(block, x, self._attention(q, k, v, predict_mask))
        x = self.reference.transformer.output_norm(x)
        return self.reference.lm_head(x).reshape(b, s, n, -1)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.forward_selected(input_ids, torch.arange(input_ids.shape[1], device=input_ids.device))

    def forward_selected(self, input_ids: Tensor, positions: Tensor) -> Tensor:
        cache, docs = self._encode_real(input_ids, None)
        positions = self._positions(positions, input_ids.shape[0], cache.length, input_ids.device)
        return self._predict(cache, docs, positions)

    def encode(self, input_ids: Tensor, cache: MTPCache | None = None, *, positions=None):
        """Append real tokens; optionally project only selected absolute prefixes."""
        previous = 0 if cache is None else cache.length
        cache, docs = self._encode_real(input_ids, cache)
        if positions is None:
            positions = torch.arange(previous, cache.length, device=input_ids.device)
        positions = self._positions(positions, input_ids.shape[0], cache.length, input_ids.device)
        return cache, self._predict(cache, docs, positions)
