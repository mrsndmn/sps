"""Four-GPU CE training of frozen Medusa-style heads or a small causal draft."""
import argparse,contextlib,json,math,os,time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import experiments as e
from scripts.mtp_common import TokenSequences,sequence_starts
from scripts.spec_baseline_common import construct,identities,restore,trainable_state,loss_for,validate_resume,finalize_checkpoint
from scripts.pilot_run import write_json


@torch.no_grad()
def validation(model,exp,device):
    data=TokenSequences('val',e.BASELINE_CONTEXT,e.BASELINE_RESERVE_HORIZONS)
    total=0.;count=0;model.eval()
    for start in range(0,e.BASELINE_VAL_TOKENS,e.BASELINE_CONTEXT):
        tokens=data[start][None].to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=loss_for(model(tokens[:,:e.BASELINE_CONTEXT]),tokens,exp.recipe,model.config)
        total+=float(loss);count+=1
    model.train();return total/count


def main():
    p=argparse.ArgumentParser();p.add_argument('--filter',required=True);p.add_argument('--probe',action='store_true');a=p.parse_args()
    if e.SUITE!='spec-baselines':raise RuntimeError('Select spec-baselines')
    matches=e.collect_experiments(a.filter)
    if len(matches)!=1 or not matches[0].trainable:raise ValueError('Select one trainable baseline')
    exp=matches[0];rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=exp.num_gpus:raise RuntimeError('Expected four ranks')
    torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl')
    torch.manual_seed(e.MTP_SEED)
    model=construct(exp,device);params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=e.BASELINE_LR[exp.recipe],betas=e.BASELINE_BETAS,
        weight_decay=e.BASELINE_WEIGHT_DECAY)
    dest=(e.SANITY_ROOT if a.probe else e.RUN_ROOT)/'out'/exp.name;dest.mkdir(parents=True,exist_ok=True)
    signature=e.design_signature();identity=identities();step0=0
    if not a.probe and (dest/'latest.pt').exists():
        state=torch.load(dest/'latest.pt',map_location=device,weights_only=False)
        step0=validate_resume(state,signature,identity)
        restore(model,state,exp);optimizer.load_state_dict(state['optimizer'])
    steps=e.BASELINE_PROBE_STEPS if a.probe else e.BASELINE_STEPS
    data=TokenSequences('train',e.BASELINE_CONTEXT,e.BASELINE_RESERVE_HORIZONS)
    sampler=sequence_starts(len(data.tokens),e.BASELINE_CONTEXT,world,rank,
        resume=step0*e.BASELINE_GLOBAL_BATCH//world,reserve_horizons=e.BASELINE_RESERVE_HORIZONS)
    if len(sampler)<(steps-step0)*e.BASELINE_GLOBAL_BATCH//world:raise RuntimeError('Insufficient unique training data')
    batches=iter(DataLoader(data,batch_size=e.BASELINE_MICRO_BATCH,sampler=sampler,drop_last=True))
    accumulation=e.BASELINE_GLOBAL_BATCH//(world*e.BASELINE_MICRO_BATCH)
    if accumulation*world*e.BASELINE_MICRO_BATCH!=e.BASELINE_GLOBAL_BATCH:raise RuntimeError('Invalid batch sharding')
    ddp=DDP(model,device_ids=[local],broadcast_buffers=False)
    def record(row):
        if rank==0:
            with (dest/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    def save(step,name):
        path=dest/name;tmp=path.with_suffix('.tmp')
        torch.save(dict(recipe=exp.recipe,model=trainable_state(model,exp.recipe),optimizer=optimizer.state_dict(),
            step=step,tokens=step*e.BASELINE_STEP_TOKENS,design_signature=signature,**identity),tmp)
        tmp.replace(path)
    if not a.probe and step0==0:
        if rank==0:record(dict(event='validation',step=0,ce=validation(model,exp,device)))
        dist.barrier()
    model.train();start=time.monotonic()
    for step in range(step0,steps):
        torch.cuda.synchronize();began=time.monotonic()
        factor=min(1.,(step+1)/e.BASELINE_WARMUP)*(.1+.9*(1+math.cos(math.pi*max(0,step-e.BASELINE_WARMUP)/max(1,e.BASELINE_STEPS-e.BASELINE_WARMUP)))/2)
        for group in optimizer.param_groups:group['lr']=e.BASELINE_LR[exp.recipe]*factor
        optimizer.zero_grad(set_to_none=True);total=0.
        for micro in range(accumulation):
            tokens=next(batches).to(device)
            sync=contextlib.nullcontext() if micro==accumulation-1 else ddp.no_sync()
            with sync,torch.autocast('cuda',dtype=torch.bfloat16):
                logits=ddp(tokens[:,:e.BASELINE_CONTEXT]);loss=loss_for(logits,tokens,exp.recipe,model.config)/accumulation
                loss.backward()
            total+=float(loss.detach());del logits,loss
        norm=torch.nn.utils.clip_grad_norm_(params,e.BASELINE_CLIP)
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite baseline gradient')
        optimizer.step();torch.cuda.synchronize()
        record(dict(event='train',step=step+1,tokens=(step+1)*e.BASELINE_STEP_TOKENS,
            step_seconds=time.monotonic()-began,loss=total,lr=optimizer.param_groups[0]['lr']))
        if not a.probe and ((step+1)%e.BASELINE_EVAL_INTERVAL==0 or step+1==steps):
            dist.barrier()
            if rank==0:
                record(dict(event='validation',step=step+1,ce=validation(model,exp,device)))
                save(step+1,'latest.pt')
            dist.barrier()
        if time.monotonic()-start>e.BASELINE_MAX_HOURS*3600:raise TimeoutError('Baseline training time cap')
    if not a.probe and rank==0:finalize_checkpoint(dest,exp,signature,identity)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
