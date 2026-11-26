"""Train a DFlash-b16 drafter from random weights on the prepared 100K corpus."""

from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import json
import os
import random
from statistics import mean, median
import tempfile
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint

from workloads.distillation import dflash_mask_ablation as reference
from data import instruct_data
from orchestration import mlspace_launch as launch
from orchestration.named.publication import work_path
from workloads.distillation import qwen4b_drafter_data as prepared, qwen4b_drafter_train as previous
from experiments.dflash_repro_train import TrainArguments, experiment_configs



def training_settings(experiment):
    settings = asdict(experiment)
    for key in ("output_root", "eval_dir_pattern"):
        settings.pop(key, None)
    return settings


def training_inputs(experiment):
    return dict(settings=training_settings(experiment), **input_provenance())


def input_provenance():
    return dict(
        selection_sha256=instruct_data.file_hash(previous.selection_path()),
        target_revision=reference.TARGET_REVISION, draft_revision=reference.DRAFT_REVISION,
    )


def total_steps(row_count, exp):
    if row_count % exp.global_batch or exp.global_batch % exp.num_gpus:
        raise ValueError("DFlash training requires complete global batches")
    return row_count // exp.global_batch * exp.epochs


def epoch_order(indices, epoch, experiment):
    order = list(indices)
    random.Random(experiment.seed + epoch).shuffle(order)
    return order


def sample_anchors(mask, epoch, row_id, experiment):
    eligible = [i for i in range(1, len(mask) - experiment.proposals)
                if mask[i] and any(mask[i + 1 : i + experiment.width])]
    if not eligible:
        raise ValueError(f"No supervised DFlash anchors in row {row_id}")
    sampler = random.Random(f"{experiment.seed}:{epoch}:{row_id}")
    return sorted(sampler.sample(eligible, min(experiment.max_anchors, len(eligible))))


def block_visibility(starts, source_width, q, kv, experiment):
    group = q // experiment.width
    return (kv < starts[group]) | ((kv >= source_width) & ((kv - source_width) // experiment.width == group))


class BlockLoss(nn.Module):
    def __init__(self, draft, head, experiment):
        super().__init__()
        self.experiment = experiment
        self.draft = draft
        # The target owns the head. Avoid including its frozen weights in DDP/checkpoints.
        self.head_weight = head.weight.detach()

    def forward(self, noise, context, positions, mask, labels, weights):
        hidden = self.draft(
            noise_embedding=noise, target_hidden=context, position_ids=positions,
            attention_mask=mask, is_causal=False,
        )[0]
        hidden = hidden.view(-1, self.experiment.width, hidden.size(-1))[:labels.size(0), 1:].reshape(-1, hidden.size(-1))
        labels, weights = labels.flatten(), weights.flatten()

        def loss_chunk(features, gold, scale):
            logits = F.linear(features, self.head_weight)
            return (F.cross_entropy(logits.float(), gold, reduction="none") * scale).sum()

        # Recompute head logits per chunk in backward to bound vocabulary activation memory.
        parts = [checkpoint(loss_chunk, hidden[i : i + self.experiment.head_chunk], labels[i : i + self.experiment.head_chunk],
                            weights[i : i + self.experiment.head_chunk], use_reentrant=False, preserve_rng_state=False)
                 for i in range(0, labels.numel(), self.experiment.head_chunk)]
        return torch.stack(parts).sum() / weights.sum()


def build_draft(device, *, backend="sdpa", experiment):
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    pinned = dict(revision=reference.DRAFT_REVISION, local_files_only=True)
    config = AutoConfig.from_pretrained(reference.DRAFT, **pinned)
    if config.block_size != experiment.width or config.num_hidden_layers != 5 or config.attention_dropout:
        raise RuntimeError("Unexpected architecture in the pinned DFlash config")
    cls = get_class_from_dynamic_module(config.auto_map["AutoModel"], reference.DRAFT, **pinned)
    if backend == "flex":
        compiled = torch.compile(flex_attention, dynamic=True, fullgraph=True)

        def attention(module, q, k, v, attention_mask, scaling, **kwargs):
            result = compiled(q.to(v.dtype), k.to(v.dtype), v, block_mask=attention_mask,
                              enable_gqa=True, scale=scaling)
            return result.transpose(1, 2).contiguous(), None

        ALL_ATTENTION_FUNCTIONS.register("dflash_repro_flex", attention)
        backend = "dflash_repro_flex"
    # The custom implementation is selected after model construction.
    config._attn_implementation = "sdpa"
    torch.manual_seed(experiment.seed)
    model = cls(config).to(device)
    model.config._attn_implementation = backend
    if tuple(model.target_layer_ids) != previous.TARGET_LAYERS:
        raise RuntimeError("Unexpected Qwen feature layers")
    return model


def draft_inputs(target, draft, row, anchors, device, *, sparse=True, experiment):
    tokens = torch.tensor([row["input_ids"]], device=device)
    supervision = torch.tensor(row["loss_mask"], device=device, dtype=torch.bool)
    starts = torch.tensor(anchors, device=device)
    source = torch.arange(tokens.size(1), device=device)
    with torch.no_grad():
        output = target.model(tokens, use_cache=False, output_hidden_states=True)
        context = reference.context_features(output, draft.target_layer_ids)
        del output
        slots = torch.arange(experiment.width, device=device)
        label_positions = starts[:, None] + slots[1:]
        labels = tokens[0, label_positions]
        weights = supervision[label_positions] * torch.exp(-(slots[1:].float() - 1) / experiment.gamma)
        padding = (-len(anchors)) % (experiment.query_block // experiment.width) if sparse else 0
        padded_starts = torch.cat((starts, starts[:1].expand(padding)))
        block = torch.full((len(padded_starts), experiment.width), draft.mask_token_id, device=device,
                           dtype=torch.long)
        block[:, 0] = tokens[0, padded_starts]
        noise = target.model.embed_tokens(block.flatten()[None])
        positions = torch.cat((source, (padded_starts[:, None] + slots).flatten()))[None]

    source_width = starts.new_tensor(len(source))

    def mask_mod(b, h, q, kv):
        return block_visibility(padded_starts, source_width, q, kv, experiment=experiment)

    q_size, kv_size = noise.size(1), positions.size(1)
    if sparse:
        mask = create_block_mask(mask_mod, None, None, q_size, kv_size, device=device,
                                 BLOCK_SIZE=experiment.query_block, _compile=True)
    else:
        q = torch.arange(q_size, device=device)[:, None]
        kv = torch.arange(kv_size, device=device)[None, :]
        mask = mask_mod(0, 0, q, kv)[None, None]
    return noise, context, positions, mask, labels, weights


def save_checkpoint(folder, model, optimizer, schedule, step, design):
    from safetensors.torch import save_file

    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder.parent) as tmp:
        staging = Path(tmp) / "checkpoint"
        staging.mkdir()
        weights = model.state_dict()
        torch.save(dict(model=weights, optimizer=optimizer.state_dict(), scheduler=schedule.state_dict(),
                        step=step, design=design), staging / "resume.pt")
        save_file({key: value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                   for key, value in weights.items()}, staging / "weights.safetensors")
        launch.store_json(staging / "inputs.json", design)
        staging.rename(folder)


def load_checkpoint(folder, model, optimizer, schedule, design):
    state = torch.load(folder / "resume.pt", map_location="cpu", weights_only=True)
    if state["design"] != design:
        raise RuntimeError("Checkpoint belongs to a different training protocol")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    schedule.load_state_dict(state["scheduler"])
    return state["step"]


def checkpoint_path(step, experiment):
    return work_path(experiment.output("train")) / f"step-{step:05d}"


def train(probe, experiment):
    from transformers import get_cosine_schedule_with_warmup

    startup = time.monotonic()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != experiment.num_gpus:
        raise RuntimeError(f"Expected {experiment.num_gpus} DDP ranks, got {world}")
    device = torch.device("cuda", local)
    design = training_inputs(experiment)
    corpus = prepared.open_dataset()
    train_indices = previous.selection()["train"]
    steps = total_steps(len(train_indices), experiment)
    accumulation = experiment.global_batch // world
    per_epoch = len(train_indices) // experiment.global_batch
    target, _ = previous.load_target(device)
    model = build_draft(device, backend="flex", experiment=experiment)
    ddp = DistributedDataParallel(BlockLoss(model, target.lm_head, experiment=experiment), device_ids=[local],
                                 broadcast_buffers=False, gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=experiment.lr, fused=True)
    schedule = get_cosine_schedule_with_warmup(optimizer, int(steps * experiment.warmup), steps)
    start = 0
    milestone_steps = (*experiment.milestones, steps)
    if not probe:
        for at in reversed(milestone_steps):
            folder = checkpoint_path(at, experiment=experiment)
            if folder.is_dir():
                start = load_checkpoint(folder, model, optimizer, schedule, design)
                if start != at:
                    raise RuntimeError("Checkpoint step does not match its path")
                break
    durations = []
    last_epoch, order = -1, None
    stop = experiment.probe_steps if probe else steps
    if probe:
        # One full step on the longest available examples, then the real epoch order.
        lengths = corpus["seq_len"]
        stress = sorted(train_indices, key=lengths.__getitem__, reverse=True)[:experiment.global_batch]
    for step in range(start, stop):
        epoch, within_epoch = divmod(step, per_epoch)
        if epoch != last_epoch:
            order = epoch_order(train_indices, epoch, experiment=experiment)
            last_epoch = epoch
        first = within_epoch * experiment.global_batch
        batch = stress if probe and step == 0 else order[first:first + experiment.global_batch]
        began = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss_total = torch.zeros((), device=device)
        most_anchors = 0
        for micro, row_id in enumerate(batch[rank::world]):
            row = corpus[row_id]
            anchors = sample_anchors(row["loss_mask"], epoch, row_id, experiment=experiment)
            most_anchors = max(most_anchors, len(anchors))
            args = draft_inputs(target, model, row, anchors, device, experiment=experiment)
            sync = ddp.no_sync() if micro + 1 < accumulation else nullcontext()
            with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                loss = ddp(*args)
                (loss / accumulation).backward()
            loss_total += loss.detach()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
        del args
        torch.cuda.synchronize(device)
        elapsed = torch.tensor([time.monotonic() - began, most_anchors], device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        seconds, anchors_peak = elapsed.tolist()
        if rank == 0 and (probe or (step + 1) % 50 == 0):
            print(f"step={step + 1} loss_rank0={loss_total.item()/accumulation:.5f} "
                  f"grad={norm.item():.4f} seconds={seconds:.2f} anchors={int(anchors_peak)}", flush=True)
        if probe and step >= experiment.probe_warmup:
            durations.append(seconds)
        if not probe and step + 1 in milestone_steps:
            if rank == 0:
                save_checkpoint(checkpoint_path(step + 1, experiment=experiment), model, optimizer, schedule, step + 1, design)
            dist.barrier()
    peak = torch.tensor(torch.cuda.max_memory_allocated(device), device=device, dtype=torch.int64)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    if rank == 0:
        if probe:
            launch.store_json(work_path(experiment.output("probe")) / "prepared.json", dict(
                design=design, gpu=torch.cuda.get_device_name(device), world=world,
                full_steps=steps, measured_steps=len(durations), wall_seconds=time.monotonic()-startup,
                mean_step_seconds=mean(durations), median_step_seconds=median(durations),
                measured_step_seconds=durations, max_allocated_bytes=peak.item(),
                estimated_full_hours=mean(durations)*steps/3600,
            ))
        else:
            launch.store_json(checkpoint_path(steps, experiment=experiment).parent / "trained.json", dict(
                design=design, steps=steps, rows=len(train_indices), checkpoints=milestone_steps,
            ))
    dist.barrier()
    dist.destroy_process_group()
