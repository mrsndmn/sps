"""Token-normalized assistant-only DDP CE/KD adaptation."""
import argparse
import contextlib
import json
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from scripts import sched_common as sc
from scripts.sched_common import e
from scripts import instruct_config as c
from scripts.instruct_common import load_model, load_teacher, loss, nll
from scripts import sched_config as schedule
from scripts.instruct_data import rows
from scripts.mlspace_launch import store_json


def save(path, model, optimizer, step, best, stage, plan):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
        best=best, stage=stage, plan=plan, design_signature=sc.signature(), source=c.MODEL,
        revision=c.REVISION, experiment=ACTIVE_EXPERIMENT.__dict__,
        source_signature=schedule.SOURCE_SIGNATURE), temporary)
    temporary.replace(path)


def restore(path, model, device):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if (state['design_signature'] != sc.signature() or state['source'] != c.MODEL
            or state['revision'] != c.REVISION or state['experiment'] != ACTIVE_EXPERIMENT.__dict__
            or state['plan'] != sc.source_plan()):
        raise RuntimeError('Scheduled resume provenance mismatch')
    model.load_state_dict(state['model'])
    return state


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
    args = parser.parse_args()
    matches = e.collect_experiments(args.filter)
    if len(matches) != 1 or not matches[0].trainable:
        raise ValueError('Select exactly one trainable arm')
    exp = matches[0]
    global ACTIVE_EXPERIMENT
    ACTIVE_EXPERIMENT = exp
    rank = int(os.environ['RANK']); world = int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK'])
    if world != exp.num_gpus:
        raise ValueError('Unexpected world size')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl')
    plan = sc.source_plan()
    steps = plan['steps']; transition = plan['stage_steps']
    if not 0 < transition < steps <= c.MAX_STEPS:
        raise ValueError('Invalid preparation plan')
    root = sc.output_dir('cosine',exp)
    root.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        store_json(root/'lr_schedule.json',dict(design_signature=sc.signature(),scope=schedule.SCHEDULE_SCOPE,
            warmup_fraction=schedule.WARMUP_FRACTION,lrs=[sc.learning_rate(exp.recipe,i,plan) for i in range(steps)]))
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
    # Restore optimizer tensors, then release duplicate CPU checkpoint storage.
    state = None
    model.train()
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False)
    if c.MICRO_BATCH != 1:
        raise ValueError('Variable-length instruct batches require MICRO_BATCH=1')
    accumulation = c.GLOBAL_BATCH//(world*c.MICRO_BATCH)
    if accumulation*world*c.MICRO_BATCH != c.GLOBAL_BATCH:
        raise ValueError('Invalid global batch')
    interval = max(1, steps//c.CHECKPOINT_PARTS)
    for step in range(start, steps):
        if exp.recipe == 'staged_sps' and step == transition:
            del ddp
            optimizer = optimizer_for(model, False)
            ddp = DDP(model, device_ids=[local], broadcast_buffers=False)
            frozen = False
            best = float('inf')  # separate stage-two model selection
        lr = sc.learning_rate(exp.recipe, step, plan)
        for group in optimizer.param_groups:
            group['lr'] = lr
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
        milestone = step+1 == transition or (step+1) % interval == 0 or step+1 == steps
        if milestone:
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
                                                 lr=lr, train_loss=totals[0].item(), ce=totals[1].item(),
                                                 kd=totals[2].item(), supervised_tokens=denominator.item(),
                                                 distillation_weight=exp.distillation_weight))+'\n')
            dist.barrier()
        model.train()
    if rank == 0:
        sc.validate_checkpoint('cosine',exp)
        store_json(root/'trained.json',dict(design_signature=sc.signature(),experiment=exp.__dict__,plan=plan))
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    main()
