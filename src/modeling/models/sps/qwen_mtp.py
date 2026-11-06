"""Native Qwen SPS-MTP with a shared backbone and group-local prediction slots."""

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def visibility(q, k, horizons):
    width = horizons + 1
    qgroup, qslot = q // width, q % width
    kgroup, kslot = k // width, k % width
    real = (kslot == 0) & (kgroup <= qgroup)
    local = (qslot > 0) & (kslot > 0) & (qgroup == kgroup) & (kslot <= qslot)
    return real | local


def horizon_targets(tokens, assistant_mask, horizons):
    """Supervise only futures whose entire gap is assistant output in this row."""
    if tokens.shape != assistant_mask.shape or tokens.ndim != 2:
        raise ValueError("Expected aligned batched tokens and assistant mask")
    if horizons < 1:
        raise ValueError("Positive horizon count required")
    length = tokens.shape[1] - 1
    labels = tokens.new_full((*tokens.shape[:1], length, horizons), -100)
    valid = torch.ones_like(tokens, dtype=torch.bool)
    for h in range(1, horizons + 1):
        count = tokens.shape[1] - h
        if count <= 0:
            break
        valid[:, :count] &= assistant_mask[:, h:].bool()
        labels[:, :count, h - 1] = torch.where(valid[:, :count], tokens[:, h:], -100)
    return labels


class QwenSPSMTP(nn.Module):
    def __init__(self, sps, horizons, *, backend="sdpa"):
        super().__init__()
        if not sps.sps or sps.config.model_type != "qwen3" or horizons < 2:
            raise ValueError("Expected Qwen3 SPS and at least two horizons")
        if backend not in {"sdpa", "flex_attention"}:
            raise ValueError("Unsupported attention backend")
        self.backbone = sps.backbone
        self.prediction = sps.prediction
        self.new_embeddings = nn.Parameter(
            self.prediction.detach().repeat(horizons - 1, 1)
        )
        self.horizons = horizons
        self.backend = backend
        self.config = sps.config
        self.backbone.set_attn_implementation(backend)
        self.embedding_only = False

    def set_stage(self, embedding_only):
        self.embedding_only = embedding_only
        self.backbone.requires_grad_(not embedding_only)
        self.prediction.requires_grad_(not embedding_only)
        self.new_embeddings.requires_grad_(True)

    def forward(self, input_ids, *, targets=None, positions=None):
        batch, length = input_ids.shape
        width = self.horizons + 1
        device = input_ids.device
        real = self.backbone.get_input_embeddings()(input_ids)
        embeddings = torch.cat((self.prediction[None], self.new_embeddings)).to(
            real.dtype
        )
        states = torch.cat(
            (real[:, :, None], embeddings[None, None].expand(batch, length, -1, -1)), 2
        )
        states = states.flatten(1, 2)
        indices = torch.arange(length * width, device=device)
        rope_positions = (indices // width + indices % width)[None]
        if int(length + self.horizons - 1) >= self.config.max_position_embeddings:
            raise ValueError("MTP positions exceed native context")
        if self.backend == "flex_attention":
            from torch.nn.attention.flex_attention import create_block_mask

            def allowed(b, head, q, k):
                return (
                    (q < length * width)
                    & (k < length * width)
                    & visibility(q, k, self.horizons)
                )

            mask = create_block_mask(
                allowed,
                batch,
                None,
                length * width,
                length * width,
                device=str(device),
                _compile=True,
            )
        else:
            mask = visibility(indices[:, None], indices[None, :], self.horizons)[
                None, None
            ]
        hidden = self.backbone.model(
            inputs_embeds=states,
            position_ids=rope_positions,
            attention_mask={"full_attention": mask},
            use_cache=False,
        ).last_hidden_state
        hidden = hidden.reshape(batch, length, width, -1)[:, :, 1:]
        if positions is not None:
            hidden = hidden[:, positions]
        if targets is None:
            return SimpleNamespace(logits=self.backbone.lm_head(hidden))
        if targets.shape != hidden.shape[:-1]:
            raise ValueError("Targets must align with prediction slots")
        if self.embedding_only:
            targets = targets.clone()
            targets[:, :, 0] = -100
        selected = (targets != -100).flatten().nonzero().flatten()
        if not len(selected):
            raise ValueError("No supervised future tokens")
        flat = hidden.flatten(0, 2)
        labels = targets.flatten()
        summed = hidden.new_zeros((), dtype=torch.float32)

        def objective(features, gold):
            return F.cross_entropy(
                self.backbone.lm_head(features).float(), gold, reduction="sum"
            )

        for start in range(0, len(selected), 64):
            ids = selected[start : start + 64]
            value = (
                checkpoint(objective, flat[ids], labels[ids], use_reentrant=False)
                if torch.is_grad_enabled()
                else objective(flat[ids], labels[ids])
            )
            summed = summed + value
        return summed, len(selected)
