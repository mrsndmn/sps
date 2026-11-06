"""Long base CE worker; pilot DDP loop with absolute-step warmup/cosine LR."""
import argparse
import contextlib
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from scripts import base50_config as config
e, c = config.configure()
from scripts.ar_sps_common import load_model, load_teacher, loss, nll, save, restore, read_plan
from scripts.ar_sps_data import rows
from scripts.mlspace_launch import store_json


def optimizer_for(model, embedding_only):
    model.set_stage(embedding_only)
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=c.EMBED_LR if embedding_only else c.FULL_LR,
                            betas=c.BETAS, weight_decay=0. if embedding_only else c.WEIGHT_DECAY)


def finalize(root, model, optimizer, step, best, frozen, plan):
    # A resume at the last step skips the loop: always reconstruct final.pt before
    # the terminal marker, including interruption after the last latest.pt save.
    if step != plan['steps'] or frozen:
        raise RuntimeError('Cannot complete unfinished training')
    save(root/'final.pt', model, optimizer, step, best, frozen, plan)
    store_json(root/'trained.json', dict(design_signature=e.design_signature(), step=step,
                                       checkpoint=str(root/'final.pt'), reminder=config.REMINDER))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', required=True)
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
    plan = read_plan()
    steps = plan['steps']; transition = plan['stage_steps']
    root = e.RUN_ROOT/'out'/exp.name
    root.mkdir(parents=True, exist_ok=True)
    model = load_model(exp.recipe, device)
    teacher = load_teacher(device) if exp.distillation_weight else None
    data = rows('train'); validation = rows('val')
    if len(data) < steps*c.GLOBAL_BATCH:
        raise RuntimeError('Insufficient unique training rows')
    start = 0; best = float('inf'); state = None
    if (root/'latest.pt').exists():
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
    interval = config.CHECKPOINT_INTERVAL
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
        ce_total = 0.
        kd_total = 0.
        for micro in range(accumulation):
            offset = step*c.GLOBAL_BATCH + (micro*world+rank)*c.MICRO_BATCH
            batch = torch.tensor(np.asarray(data[offset:offset+c.MICRO_BATCH], dtype='int64'), device=device)
            sync = contextlib.nullcontext() if micro == accumulation-1 else ddp.no_sync()
            with sync, torch.autocast('cuda', dtype=torch.bfloat16):
                value, ce, kd = loss(ddp, batch, teacher=teacher, weight=exp.distillation_weight,
                                     temperature=c.DISTILLATION_TEMPERATURE, components=True)
                value = value/accumulation
                value.backward()
            total += value.detach().item()
            ce_total += ce.item()/accumulation
            kd_total += kd.item()/accumulation
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], c.CLIP)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite gradient')
        lr = config.learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.monotonic()-began
        if rank == 0 and (step+1) % config.LOG_INTERVAL == 0:
            print(json.dumps(dict(step=step+1, lr=lr, loss=total, seconds_per_step=elapsed)), flush=True)
        milestone = step+1 == transition or (step+1) % interval == 0 or step+1 == steps
        if milestone:
            dist.barrier()
            if rank == 0:
                score = float(np.mean(nll(model, validation, device)))
                if score < best:
                    best = score
                    save(root/('embedding_best.pt' if frozen else 'best.pt'), model, optimizer, step+1, best, frozen, plan)
                save(root/'latest.pt', model, optimizer, step+1, best, frozen, plan)
                if step+1 == transition:
                    save(root/'stage_endpoint.pt', model, optimizer, step+1, best, frozen, plan)
                with (root/'metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(step=step+1, tokens=(step+1)*c.GLOBAL_BATCH*c.CONTEXT,
                                                 val_nll=score, lr=lr, rank_zero_train_loss=total,
                                                 rank_zero_ce=ce_total, rank_zero_kd=kd_total,
                                                 distillation_weight=exp.distillation_weight))+'\n')
            dist.barrier()
        model.train()
    if rank == 0:
        finalize(root, model, optimizer, steps, best, frozen, plan)
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    main()
