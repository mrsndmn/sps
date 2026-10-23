"""DDP SPS MTP adaptation with embedding-only or full CE fine-tuning."""
import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

import experiments as e
from scripts.mtp_common import TokenSequences, load_mtp, shifted_targets, sequence_starts
from scripts.pilot_run import write_json


def adaptation_loss(logits, targets, *, all_horizons=False):
    if all_horizons:
        # Ordinary token-level CE over all valid horizons, including P1.
        return torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]).float(),
            targets.reshape(-1),ignore_index=-100,reduction='sum')/(targets!=-100).sum().clamp_min(1)
    losses=[]
    for j in range(1,logits.shape[2]):
        labels=targets[:,:,j]
        losses.append(torch.nn.functional.cross_entropy(logits[:,:,j].reshape(-1,logits.shape[-1]).float(),
            labels.reshape(-1),ignore_index=-100,reduction='sum')/(labels!=-100).sum().clamp_min(1))
    return torch.stack(losses).mean()


def optimizer_groups(model):
    if not e.MTP_UNFREEZE:
        return [dict(params=[model.new_embeddings],lr=e.MTP_LR,base_lr=e.MTP_LR,
                     weight_decay=e.MTP_WEIGHT_DECAY)]
    shared=model.reference.transformer.wte.weight
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        embedding=p is shared or p is model.new_embeddings
        lr=e.MTP_EMBED_LR if embedding else e.MTP_LR
        decay=0.0 if embedding or p.ndim<2 else e.MTP_WEIGHT_DECAY
        groups.setdefault((lr,decay),[]).append(p)
    return [dict(params=params,lr=lr,base_lr=lr,weight_decay=decay)
            for (lr,decay),params in groups.items()]


def save_checkpoint(path, model, optimizer, step, signature):
    temporary=path.with_suffix('.tmp')
    state=dict(new_embeddings=model.new_embeddings.detach().cpu(), optimizer=optimizer.state_dict(),
        step=step, tokens=step*e.MTP_STEP_TOKENS,horizons=model.horizons,
        source=str(e.MTP_SOURCE),design_signature=signature,train_backbone=e.MTP_UNFREEZE)
    if e.MTP_UNFREEZE:state['backbone']=model.reference.state_dict()
    torch.save(state,temporary)
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--filter',required=True)
    parser.add_argument('--probe',action='store_true')
    args=parser.parse_args()
    if e.SUITE not in {'mtp','mtp-unfreeze'}:raise RuntimeError('Select an MTP suite')
    matches=e.collect_experiments(args.filter)
    if len(matches)!=1:raise ValueError('Select exactly one MTP experiment')
    exp=matches[0]
    rank=int(os.environ['RANK']); world=int(os.environ['WORLD_SIZE']); local=int(os.environ['LOCAL_RANK'])
    if world!=exp.num_gpus:raise RuntimeError('Expected four ranks')
    torch.cuda.set_device(local); device=torch.device('cuda',local)
    dist.init_process_group('nccl')
    torch.manual_seed(e.MTP_SEED)
    model=load_mtp(exp.horizons,device,backend=e.MTP_BACKEND,train_backbone=e.MTP_UNFREEZE)
    if e.MTP_UNFREEZE:
        if not all(p.requires_grad for p in model.parameters()):raise RuntimeError('All draft parameters must train')
    elif {n for n,p in model.named_parameters() if p.requires_grad}!={'new_embeddings'}:
        raise RuntimeError('Only new embeddings may train')
    groups=optimizer_groups(model)
    optimizer=torch.optim.AdamW(groups,betas=e.MTP_BETAS)
    trainable=[p for p in model.parameters() if p.requires_grad]
    parent_samples=0
    if e.MTP_UNFREEZE:
        from scripts.mtp_common import warmstart_metadata
        parent_samples=warmstart_metadata(exp.horizons,world_size=world)['parent_samples_per_rank']
    dest=(e.SANITY_ROOT if args.probe else e.RUN_ROOT)/'out'/exp.name
    dest.mkdir(parents=True,exist_ok=True)
    signature=e.design_signature()
    start_step=0
    checkpoint=dest/'latest.pt'
    if checkpoint.exists() and not args.probe:
        state=torch.load(checkpoint,map_location=device,weights_only=False)
        if state['design_signature']!=signature:raise RuntimeError('Stale partial MTP checkpoint')
        from scripts.mtp_common import restore_draft
        restore_draft(model,state)
        optimizer.load_state_dict(state['optimizer']); start_step=state['step']
    data=TokenSequences('train',e.MTP_CONTEXT)
    sampler=sequence_starts(len(data.tokens),e.MTP_CONTEXT,world,rank,
        resume=parent_samples+start_step*e.MTP_GLOBAL_BATCH//world)
    steps=e.MTP_PROBE_STEPS if args.probe else e.MTP_TRAIN_STEPS
    if len(sampler)<(steps-start_step)*e.MTP_GLOBAL_BATCH//world:
        raise RuntimeError('Insufficient unique sequences for this rank')
    batches=iter(DataLoader(data,batch_size=e.MTP_MICRO_BATCH,sampler=sampler,drop_last=True))
    ddp=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False)
    accumulation=e.MTP_GLOBAL_BATCH//(world*e.MTP_MICRO_BATCH)
    if accumulation*world*e.MTP_MICRO_BATCH!=e.MTP_GLOBAL_BATCH:raise ValueError('Invalid accumulation')
    def record(row):
        if rank==0:
            with (dest/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    if not args.probe and start_step==0:
        if rank==0:
            from scripts.mtp_evaluate import horizon_metrics
            with torch.no_grad(): before=horizon_metrics(model,device)
            write_json(dest/'quality-before.json',before)
            record(dict(event='validation',step=0,tokens=0,metrics=before))
        dist.barrier()
    initial=time.monotonic()
    model.train()
    for step in range(start_step,steps):
        torch.cuda.synchronize(); started=time.monotonic()
        warmup=min(1.0,(step+1)/e.MTP_WARMUP_STEPS)
        progress=max(0,(step-e.MTP_WARMUP_STEPS)/max(1,e.MTP_TRAIN_STEPS-e.MTP_WARMUP_STEPS))
        factor=warmup*(0.1+0.9*(1+math.cos(math.pi*progress))/2)
        lr=e.MTP_LR*factor
        for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
        optimizer.zero_grad(set_to_none=True)
        total=0.0
        for micro in range(accumulation):
            tokens=next(batches).to(device)
            labels=shifted_targets(tokens,e.MTP_CONTEXT,exp.horizons,
                eos=model.config.eos_token_id,pad=model.config.pad_token_id)
            sync=contextlib.nullcontext() if micro==accumulation-1 else ddp.no_sync()
            with sync, torch.autocast('cuda',dtype=torch.bfloat16):
                logits=ddp(tokens[:,:e.MTP_CONTEXT])
                loss=adaptation_loss(logits,labels,all_horizons=e.MTP_UNFREEZE)/accumulation
                loss.backward()
            total+=float(loss.detach()); del logits,loss
        norm=torch.nn.utils.clip_grad_norm_(trainable,e.MTP_GRAD_CLIP)
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite draft gradient')
        optimizer.step()
        torch.cuda.synchronize()
        record(dict(event='train',step=step+1,tokens=(step+1)*e.MTP_STEP_TOKENS,
            step_seconds=time.monotonic()-started,loss=total,lr=lr,embedding_lr=e.MTP_EMBED_LR*factor,gradient_norm=float(norm)))
        if not args.probe and ((step+1)%e.MTP_EVAL_INTERVAL==0 or step+1==steps):
            dist.barrier()
            if rank==0:
                from scripts.mtp_evaluate import horizon_metrics
                with torch.no_grad(): metrics=horizon_metrics(model,device)
                record(dict(event='validation',step=step+1,tokens=(step+1)*e.MTP_STEP_TOKENS,metrics=metrics))
                save_checkpoint(checkpoint,model,optimizer,step+1,signature)
                if step+1==steps:
                    save_checkpoint(dest/'adaptation_final.pt',model,optimizer,step+1,signature)
                    write_json(dest/'quality-after.json',metrics)
            dist.barrier(); model.train()
        if time.monotonic()-initial>e.MAX_HOURS*3600:
            raise TimeoutError('Adaptation exceeded selected training cap')
    dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':main()
