"""Native Qwen next-token readout and ephemeral future-token slots."""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from scripts.vllm_sps_model import visibility


class RealReadoutMTP(nn.Module):
    def __init__(self, backbone, horizons, backend="sdpa"):
        super().__init__()
        self.backbone = backbone
        self.horizons = horizons
        self.backend = backend
        mean = backbone.get_input_embeddings().weight.detach().float().mean(dim=0)
        self.future = nn.Parameter(mean.repeat(horizons - 1, 1))
        self.backbone.set_attn_implementation(
            "sps_flex_attention" if backend == "flex_attention" else backend
        )
        self.set_stage(True)

    def set_stage(self, frozen):
        self.backbone.requires_grad_(not frozen)
        self.embedding_only = frozen

    def forward(self, ids, *, targets=None, positions=None):
        b, length = ids.shape
        width = self.horizons
        real = self.backbone.get_input_embeddings()(ids)
        slots = self.future.to(real.dtype).expand(b, length, -1, -1)
        inputs = torch.cat((real.unsqueeze(2), slots), dim=2).flatten(1, 2)
        indices = torch.arange(length * width, device=ids.device)
        slot = indices % width
        # Physical slot 1 predicts horizon 2, preserving its B-arm RoPE position.
        pos = indices // width + slot + (slot > 0).long()
        if self.backend == "flex_attention":
            from torch.nn.attention.flex_attention import create_block_mask

            def allowed(batch, head, q, k):
                return ((q < length * width) & (k < length * width)
                        & visibility(q, k, width - 1))

            mask = create_block_mask(allowed, b, None, length * width, length * width,
                                     device=str(ids.device), _compile=True)
        else:
            mask = visibility(indices[:, None], indices[None, :], width - 1)[None, None]
        hidden = self.backbone.model(inputs_embeds=inputs, position_ids=pos[None],
                                     attention_mask={"full_attention": mask},
                                     use_cache=False).last_hidden_state.reshape(b, length, width, -1)
        if positions is not None:
            hidden = hidden[:, positions]
        if targets is None:
            return SimpleNamespace(logits=self.backbone.lm_head(hidden))
        if self.embedding_only:
            hidden, targets = hidden[:, :, 1:], targets[:, :, 1:]
        selected = (targets != -100).flatten().nonzero().flatten()
        flat = hidden.reshape(-1, hidden.shape[-1])
        labels = targets.reshape(-1)

        def objective(features, gold):
            return F.cross_entropy(self.backbone.lm_head(features).float(), gold, reduction="sum")

        loss = flat.new_zeros((), dtype=torch.float32)
        for chunk in selected.split(64):
            loss = loss + checkpoint(objective, flat[chunk], labels[chunk], use_reentrant=False)
        return loss, len(selected)
