"""Matched-exposure DDP CE adaptation, including a frozen-to-full transition."""
import argparse
import contextlib
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import experiments as e
from scripts import ar_sps_config as c
from scripts.ar_sps_common import load_model, loss, nll, save, restore, read_plan
from scripts.ar_sps_data import rows
from scripts.mlspace_launch import store_json


def optimizer_for(model, embedding_only):
    model.set_stage(embedding_only)
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=c.EMBED_LR if embedding_only else c.FULL_LR,
                            betas=c.BETAS, weight_decay=0. if embedding_only else c.WEIGHT_DECAY)


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
    root = (e.SANITY_ROOT if args.probe else e.RUN_ROOT)/'out'/exp.name
    root.mkdir(parents=True, exist_ok=True)
    model = load_model(exp.recipe, device)
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
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False)
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
        total = 0.
        for micro in range(accumulation):
            offset = step*c.GLOBAL_BATCH + (micro*world+rank)*c.MICRO_BATCH
            batch = torch.tensor(np.asarray(data[offset:offset+c.MICRO_BATCH], dtype='int64'), device=device)
            sync = contextlib.nullcontext() if micro == accumulation-1 else ddp.no_sync()
            with sync, torch.autocast('cuda', dtype=torch.bfloat16):
                value = loss(ddp, batch)/accumulation
                value.backward()
            total += value.detach().item()
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], c.CLIP)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite gradient')
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.monotonic()-began
        phase = 'embedding' if frozen else 'full'
        if step >= c.WARMUP_PROBE and step != transition:
            times[phase].append(elapsed)
        milestone = step+1 == transition or (step+1) % interval == 0 or step+1 == steps
        if not args.probe and milestone:
            dist.barrier()
            if rank == 0:
                score = float(np.mean(nll(model, validation, device)))
                if score < best:
                    best = score
                    save(root/('embedding_best.pt' if frozen else 'best.pt'), model, optimizer, step+1, best, frozen, plan)
                save(root/'latest.pt', model, optimizer, step+1, best, frozen, plan)
                if step+1 == transition:
                    save(root/'stage_endpoint.pt', model, optimizer, step+1, best, frozen, plan)
                if step+1 == steps:
                    save(root/'final.pt', model, optimizer, step+1, best, frozen, plan)
                with (root/'metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(step=step+1, tokens=(step+1)*c.GLOBAL_BATCH*c.CONTEXT,
                                                 val_nll=score, rank_zero_train_loss=total))+'\n')
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
