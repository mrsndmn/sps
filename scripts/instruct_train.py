"""Token-normalized assistant-only DDP CE/KD adaptation."""
import argparse
import contextlib
import json
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import experiments as e
from scripts import instruct_config as c
from scripts.instruct_common import load_model, load_teacher, loss, nll, save, restore, read_plan
from scripts.instruct_data import rows
from scripts.mlspace_launch import store_json


def optimizer_for(model, embedding_only):
    model.set_stage(embedding_only)
    if not embedding_only:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=c.EMBED_LR if embedding_only else c.FULL_LR,
                            betas=c.BETAS, weight_decay=0. if embedding_only else c.WEIGHT_DECAY)


def save_progress(root, model, optimizer, step, best, frozen, plan):
    # Publish the resumable progress cursor only after its milestone artifacts.
    # A failed final save must leave latest at the previous completed interval.
    if step == plan['stage_steps']:
        save(root/'stage_endpoint.pt', model, optimizer, step, best, frozen, plan)
    if step == plan['steps']:
        save(root/'final.pt', model, optimizer, step, best, frozen, plan)
    save(root/'latest.pt', model, optimizer, step, best, frozen, plan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', required=True)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    matches = e.collect_experiments(args.filter)
    if len(matches) != 1 or not matches[0].trainable:
        raise ValueError('Select exactly one trainable arm')
    exp = matches[0]
    rank = int(os.environ['RANK']); world = int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK'])
    if world != exp.num_gpus:
        raise ValueError('Unexpected world size')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl')
    plan = dict(steps=c.PROBE_STEPS, stage_steps=c.PROBE_STEPS//2) if args.probe else read_plan()
    steps = plan['steps']; transition = plan['stage_steps']
    if not 0 < transition < steps <= c.MAX_STEPS:
        raise ValueError('Invalid preparation plan')
    root = (e.SANITY_ROOT if args.probe else e.RUN_ROOT)/'out'/exp.name
    root.mkdir(parents=True, exist_ok=True)
    model = load_model(exp.recipe, device)
    teacher = load_teacher(device) if exp.distillation_weight else None
    data = rows('train'); validation = rows('val')
    if len(data) < steps*c.GLOBAL_BATCH:
        raise RuntimeError('Insufficient unique training rows')
    start = 0; best = float('inf'); state = None
    if not args.probe and (root/'latest.pt').exists():
        state = restore(root/'latest.pt', model, device)
        if state['plan'] != plan:
            raise RuntimeError('Resume plan mismatch')
        start = state['step']; best = state['best']
    frozen = exp.recipe == 'staged_sps' and start < transition
    optimizer = optimizer_for(model, frozen)
    if state and state['stage'] == frozen:
        optimizer.load_state_dict(state['optimizer'])
    # Restore optimizer tensors, then release duplicate CPU checkpoint storage.
    state = None
    model.train()
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False)
    if c.MICRO_BATCH != 1:
        raise ValueError('Variable-length instruct batches require MICRO_BATCH=1')
    accumulation = c.GLOBAL_BATCH//(world*c.MICRO_BATCH)
    if accumulation*world*c.MICRO_BATCH != c.GLOBAL_BATCH:
        raise ValueError('Invalid global batch')
    times = {'embedding': [], 'full': []}
    interval = max(1, steps//c.CHECKPOINT_PARTS)
    for step in range(start, steps):
        if exp.recipe == 'staged_sps' and step == transition:
            del ddp
            optimizer = optimizer_for(model, False)
            ddp = DDP(model, device_ids=[local], broadcast_buffers=False)
            frozen = False
            best = float('inf')  # separate stage-two model selection
        torch.cuda.synchronize(); began = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        # DDP averages gradients across ranks: scale local summed token loss
        # by world/global supervised-token count, not by rank or example means.
        batch_rows = [data[step*c.GLOBAL_BATCH + micro*world+rank]
                      for micro in range(accumulation)]
        denominator = torch.tensor(sum(sum(row['loss_mask'][1:]) for row in batch_rows),
                                   dtype=torch.float64, device=device)
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
        if denominator.item() <= 0:
            raise RuntimeError('Empty global supervised batch')
        totals = torch.zeros(3, dtype=torch.float64, device=device)
        for micro, row in enumerate(batch_rows):
            sync = contextlib.nullcontext() if micro == accumulation-1 else ddp.no_sync()
            with sync, torch.autocast('cuda', dtype=torch.bfloat16):
                value, ce, kd, _ = loss(ddp, row, device, teacher=teacher,
                                        weight=exp.distillation_weight,
                                        temperature=c.DISTILLATION_TEMPERATURE)
                (value * (world/denominator)).backward()
            totals += torch.stack((value.detach(), ce, kd)).double()
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        totals /= denominator
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], c.CLIP)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite gradient')
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.monotonic()-began
        phase = 'embedding' if frozen else 'full'
        if step >= c.WARMUP_PROBE and not (exp.recipe == 'staged_sps' and transition <= step < transition+c.WARMUP_PROBE):
            times[phase].append(elapsed)
        milestone = step+1 == transition or (step+1) % interval == 0 or step+1 == steps
        if not args.probe and milestone:
            dist.barrier()
            if rank == 0:
                validation_result = nll(model, validation, device)
                score = validation_result['nll']
                if score < best:
                    best = score
                    save(root/('embedding_best.pt' if frozen else 'best.pt'), model, optimizer, step+1, best, frozen, plan)
                save_progress(root, model, optimizer, step+1, best, frozen, plan)
                with (root/'metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(step=step+1, examples=(step+1)*c.GLOBAL_BATCH,
                                                 val_nll=score, val_by_mode=validation_result['by_mode'],
                                                 train_loss=totals[0].item(), ce=totals[1].item(),
                                                 kd=totals[2].item(), supervised_tokens=denominator.item(),
                                                 distillation_weight=exp.distillation_weight))+'\n')
            dist.barrier()
        model.train()
    if args.probe:
        # Max rank duration, not rank-zero optimism.
        values = torch.tensor([max(times[k], default=0.) for k in ('embedding', 'full')], device=device)
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        if rank == 0:
            store_json(root/'probe.json', dict(embedding=values[0].item(), full=values[1].item(),
                                              design_signature=e.design_signature()))
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    main()
