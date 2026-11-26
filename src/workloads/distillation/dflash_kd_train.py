"""Soft Qwen-to-DFlash distillation with the hard run's training schedule."""

from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from statistics import mean, median
import os
import time

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint

from experiments.environment import COMMON_ROOT
from orchestration.design_compatibility import canonical_signature
from orchestration.source_compatibility import source_hash
from workloads.distillation import dflash_repro_train as hard
from orchestration import mlspace_launch as launch
from orchestration.named.publication import work_path
from experiments.dflash_kd_train import (
    ROOT,
    TEMPERATURE,
    experiment_configs,
    EXPERIMENT,
)


def signature():
    base = hard.signature().encode()
    return canonical_signature(sha256(base + bytes.fromhex(source_hash(__file__))).hexdigest())


def run_root():
    return EXPERIMENT.output_root


def checkpoint_path(step):
    return work_path(EXPERIMENT.output("train")) / f"step-{step:05d}"


def teacher_indices(anchors, weights, device):
    positions = torch.tensor(anchors, device=device)[:, None]
    positions = positions + torch.arange(hard.PROPOSALS, device=device)
    selected = torch.nonzero(weights.flatten() > 0, as_tuple=False).flatten()
    distinct, inverse = positions.flatten()[selected].unique(sorted=True, return_inverse=True)
    return selected, distinct, inverse


@torch.no_grad()
def soft_inputs(target, draft, row, anchors, device, *, sparse=True):
    def capture(_, _args, output):
        captured.append(output.last_hidden_state)

    captured = []
    with target.model.register_forward_hook(capture):
        noise, context, positions, mask, _, weights = hard.draft_inputs(
            target, draft, row, anchors, device, sparse=sparse,
        )
    hidden, = captured
    selected, distinct, inverse = teacher_indices(anchors, weights, device)
    teacher = torch.empty((distinct.numel(), target.lm_head.weight.size(0)),
                          dtype=torch.float32, device=device)
    for start in range(0, distinct.numel(), hard.HEAD_CHUNK):
        indices = distinct[start : start + hard.HEAD_CHUNK]
        logits = target.lm_head(hidden[0, indices]).float()
        teacher[start : start + len(indices)].copy_(F.log_softmax(logits / TEMPERATURE, dim=-1))
    return noise, context, positions, mask, weights, selected, teacher, inverse


def weighted_kl(logits, teacher_logp, weights):
    student_logp = F.log_softmax(logits.float() / TEMPERATURE, dim=-1)
    per_slot = F.kl_div(student_logp, teacher_logp, log_target=True, reduction="none").sum(dim=-1)
    return (per_slot * weights).sum() * TEMPERATURE**2


class SoftBlockLoss(hard.BlockLoss):
    def forward(self, noise, context, positions, mask, weights, selected, teacher, inverse):
        hidden = self.draft(
            noise_embedding=noise, target_hidden=context, position_ids=positions,
            attention_mask=mask, is_causal=False,
        )[0]
        hidden = hidden.view(-1, hard.WIDTH, hidden.size(-1))[:weights.size(0), 1:]
        features = hidden.reshape(-1, hidden.size(-1))[selected]
        weights = weights.flatten()[selected]

        def chunk_loss(features, rows, scale):
            logits = F.linear(features, self.head_weight)
            return weighted_kl(logits, teacher[rows], scale)

        parts = []
        for start in range(0, len(selected), hard.HEAD_CHUNK):
            chunk = slice(start, start + hard.HEAD_CHUNK)
            parts.append(checkpoint(chunk_loss, features[chunk], inverse[chunk], weights[chunk],
                                    use_reentrant=False, preserve_rng_state=False))
        return torch.stack(parts).sum() / weights.sum()


def initial_hash(model):
    digest = sha256()
    for value in model.state_dict().values():
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train(probe):
    from transformers import get_cosine_schedule_with_warmup

    startup = time.monotonic()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != EXPERIMENT.num_gpus:
        raise RuntimeError(f"Expected {EXPERIMENT.num_gpus} ranks, got {world}")
    device = torch.device("cuda", local)
    design = signature()
    corpus = hard.prepared.open_dataset()
    train_indices = hard.previous.selection()["train"]
    steps = hard.total_steps(len(train_indices), EXPERIMENT)
    accumulation = EXPERIMENT.global_batch // world
    per_epoch = len(train_indices) // EXPERIMENT.global_batch
    target, _ = hard.previous.load_target(device)
    model = hard.build_draft(device, backend="flex")
    initial = initial_hash(model) if rank == 0 else None
    ddp = DistributedDataParallel(SoftBlockLoss(model, target.lm_head), device_ids=[local],
                                 broadcast_buffers=False, gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hard.LR, fused=True)
    schedule = get_cosine_schedule_with_warmup(optimizer, int(steps * hard.WARMUP), steps)
    start = 0
    milestone_steps = (*hard.MILESTONES, steps)
    if not probe:
        for at in reversed(milestone_steps):
            folder = checkpoint_path(at)
            if folder.is_dir():
                start = hard.load_checkpoint(folder, model, optimizer, schedule, design)
                if start != at:
                    raise RuntimeError("Checkpoint step does not match its path")
                break
    else:
        lengths = corpus["seq_len"]
        stress = sorted(train_indices, key=lengths.__getitem__, reverse=True)[:EXPERIMENT.global_batch]
    durations = []
    last_epoch, order = -1, None
    for step in range(start, hard.PROBE_STEPS if probe else steps):
        epoch, within_epoch = divmod(step, per_epoch)
        if epoch != last_epoch:
            order = hard.epoch_order(train_indices, epoch)
            last_epoch = epoch
        first = within_epoch * EXPERIMENT.global_batch
        batch = stress if probe and step == 0 else order[first:first + EXPERIMENT.global_batch]
        began = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss_total = torch.zeros((), device=device)
        most_anchors = 0
        for micro, row_id in enumerate(batch[rank::world]):
            row = corpus[row_id]
            anchors = hard.sample_anchors(row["loss_mask"], epoch, row_id)
            most_anchors = max(most_anchors, len(anchors))
            args = soft_inputs(target, model, row, anchors, device)
            sync = ddp.no_sync() if micro + 1 < accumulation else nullcontext()
            with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                loss = ddp(*args)
                (loss / accumulation).backward()
            loss_total += loss.detach()
            del args
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
        torch.cuda.synchronize(device)
        elapsed = torch.tensor([time.monotonic() - began, most_anchors], device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        seconds, anchors_peak = elapsed.tolist()
        if rank == 0 and (probe or (step + 1) % 50 == 0):
            print(f"step={step + 1} kl_rank0={loss_total.item()/accumulation:.5f} "
                  f"grad={norm.item():.4f} seconds={seconds:.2f} anchors={int(anchors_peak)}", flush=True)
        if probe and step >= hard.PROBE_WARMUP:
            durations.append(seconds)
        if not probe and step + 1 in milestone_steps:
            if rank == 0:
                hard.save_checkpoint(checkpoint_path(step + 1), model, optimizer, schedule, step + 1, design)
            dist.barrier()
    peak = torch.tensor(torch.cuda.max_memory_allocated(device), device=device, dtype=torch.int64)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    if rank == 0:
        if probe:
            launch.store_json(work_path(EXPERIMENT.output("probe")) / "prepared.json", dict(
                design=design, hard_design=hard.signature(), initial_sha256=initial,
                gpu=torch.cuda.get_device_name(device), world=world, full_steps=steps,
                measured_steps=len(durations), wall_seconds=time.monotonic()-startup,
                mean_step_seconds=mean(durations), median_step_seconds=median(durations),
                measured_step_seconds=durations, max_allocated_bytes=peak.item(),
                estimated_full_hours=mean(durations)*steps/3600,
            ))
        else:
            launch.store_json(checkpoint_path(steps).parent / "trained.json", dict(
                design=design, hard_design=hard.signature(), initial_sha256=initial,
                steps=steps, rows=len(train_indices), checkpoints=milestone_steps,
            ))
    dist.barrier()
    dist.destroy_process_group()
