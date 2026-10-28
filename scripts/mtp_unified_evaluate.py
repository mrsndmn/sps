"""Single-checkpoint self-speculation, paired with its own native P1 generator."""
import argparse,json,signal,time
import torch
import experiments as e
from scripts.mtp_common import load_mtp,TokenSequences,validation_starts,file_sha256
from scripts.spec_baseline_common import checked_state,identities
from scripts.mtp_evaluate import probabilities,benchmark_reference,NativePrefixVerifier
from scripts.spec_baseline_evaluate import summarize
from scripts.pilot_run import write_json
from modeling.models.sps.unified import UnifiedSPS,unified_block
from modeling.mtp_speculative import speculative_generate


@torch.inference_mode()
def generate(model,prefix,budget,mode,seed):
    engine=UnifiedSPS(model);generator=torch.Generator(device=prefix.device).manual_seed(seed)
    p=probabilities(engine.ensure(prefix)[0],model.config)
    if budget==0:return prefix.new_empty(0),None,engine,0.,0.
    anchor=p.argmax() if mode=='greedy' else torch.multinomial(p,1,generator=generator)[0]
    events=[]
    def timed(kind,fn,*args):
        if prefix.is_cuda:
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();value=fn(*args);end.record();events.append((kind,start,end));return value
        return fn(*args)
    result=None
    if int(anchor)!=model.config.eos_token_id and budget>1:
        result=speculative_generate(torch.cat((prefix,anchor[None])),
            lambda h:timed('proposal',lambda:probabilities(engine.propose(h),model.config)),
            lambda h,d:timed('verification',lambda:probabilities(engine.verify_logits(h,d),model.config)),
            max_new_tokens=budget-1,mode=mode,eos_token_id=model.config.eos_token_id,generator=generator)
        ids=torch.cat((anchor[None],result.token_ids))
        # Keep the final emitted token pending; discarded speculative KV is removed.
        engine.ensure(torch.cat((prefix,ids))[:-1])
    else:ids=anchor[None]
    if prefix.is_cuda:torch.cuda.synchronize()
    times={kind:sum(s.elapsed_time(t)/1000 for k,s,t in events if k==kind)
           for kind in ('proposal','verification')}
    if engine.prefills!=1 or engine.block_calls!=(result.rounds if result else 0):
        raise RuntimeError('Unexpected extra backbone pass or cache prefill')
    return ids,result,engine,times['proposal'],times['verification']


@torch.inference_mode()
def benchmark(model,prefix,mode,seed):
    torch.cuda.synchronize();start=time.perf_counter()
    ids,result,engine,proposal,verification=generate(model,prefix,e.UNIFIED_NEW_TOKENS,mode,seed)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    n=model.horizons-1
    return dict(token_ids=ids.tolist(),tokens=ids.numel(),initial_target_tokens=1,total_seconds=elapsed,
        rounds=result.rounds if result else 0,accepted_by_horizon=list(result.accepted_by_horizon) if result else [0]*n,
        proposed_by_horizon=list(result.proposed_by_horizon) if result else [0]*n,
        reached_by_horizon=list(result.reached_by_horizon) if result else [0]*n,
        proposal_stream_seconds=proposal,verification_stream_seconds=verification,
        fused_block_calls=engine.block_calls,prefills=engine.prefills)


@torch.inference_mode()
def audit(model,prompts):
    maximum=0.;logit_error=0.;mismatches=0;rows=0;horizon_error=0.
    for prefix in prompts[:e.MTP_PARALLEL_AUDIT_PROMPTS]:
        engine=UnifiedSPS(model);engine.ensure(prefix)
        draft=prefix.new_tensor([262,model.config.eos_token_id,198,257])
        cache,got=unified_block(model,engine.cache,draft)
        ids=torch.cat((prefix,draft))
        _,want=model.encode(ids[None],positions=torch.arange(prefix.numel(),ids.numel(),device=prefix.device))
        horizon_error=max(horizon_error,float((probabilities(got,model.config)-probabilities(want[0],model.config)).abs().max()))
        native=NativePrefixVerifier(model.reference,e.UNIFIED_NEW_TOKENS)
        native.ensure(prefix)
        # The returned rows include the initial prefix; fused rows start after draft[0].
        expected=native.verify_logits(prefix,draft)[1:]
        p=probabilities(expected,model.config);q=probabilities(got[:,0],model.config)
        maximum=max(maximum,float((p-q).abs().max()));logit_error=max(logit_error,float((expected-got[:,0]).abs().max()))
        mismatches+=int((p.argmax(-1)!=q.argmax(-1)).sum());rows+=got.shape[0]
    if max(maximum,horizon_error)>e.MTP_PARALLEL_PROB_ATOL:raise RuntimeError('Unified numerical audit failed')
    return dict(max_probability_error=maximum,max_logit_error=logit_error,argmax_mismatches=mismatches,
        audited_rows=rows,all_horizon_probability_error=horizon_error,bitwise_native_equivalent=False)


def evaluate(exp):
    if torch.cuda.device_count()!=e.EVAL_NUM_GPUS:raise RuntimeError('Expected one visible evaluation GPU')
    trained,_=checked_state(exp);checkpoint=trained['checkpoint']
    model=load_mtp(exp.horizons,'cuda',checkpoint,backend='sdpa').eval()
    data=TokenSequences('val',e.UNIFIED_PROMPT_TOKENS,exp.horizons)
    starts=validation_starts(len(data.tokens),e.UNIFIED_PROMPT_TOKENS,e.UNIFIED_PROMPTS)
    prompts=[data[i][:e.UNIFIED_PROMPT_TOKENS].cuda() for i in starts]
    report=dict(name=exp.name,design_signature=e.design_signature(),**identities(),checkpoint=checkpoint,
        checkpoint_sha256=file_sha256(checkpoint),target='P1 of the same adapted checkpoint',
        prediction_horizons=exp.horizons,proposal_horizons=list(range(2,exp.horizons+1)),
        numerical_audit=audit(model,prompts),baseline={},variants={},
        settings=dict(prompts=e.UNIFIED_PROMPTS,prompt_tokens=e.UNIFIED_PROMPT_TOKENS,
            new_tokens=e.UNIFIED_NEW_TOKENS,warmup_passes=e.UNIFIED_WARMUP_PASSES,seed=e.MTP_SEED),
        limitations=['Target differs from the original SPS-10B after full continuation.',
            'Block and native floating point outputs need not match bitwise.',
            'One measured workload per mode after full-workload warmup; repeatability interval not estimated.'])
    report['variants'][str(exp.horizons-1)]={}
    for mode in ('greedy','sample'):
        for _ in range(e.UNIFIED_WARMUP_PASSES):
            for i,p in enumerate(prompts):
                benchmark_reference(model.reference,p,mode=mode,seed=e.MTP_SEED+i,max_new_tokens=e.UNIFIED_NEW_TOKENS)
                benchmark(model,p,mode,e.MTP_SEED+i)
        base=[benchmark_reference(model.reference,p,mode=mode,seed=e.MTP_SEED+i,max_new_tokens=e.UNIFIED_NEW_TOKENS)
              for i,p in enumerate(prompts)]
        runs=[benchmark(model,p,mode,e.MTP_SEED+i) for i,p in enumerate(prompts)]
        report['baseline'][mode]=base
        report['variants'][str(exp.horizons-1)][mode]=summarize(runs,base,exp.horizons-1,mode)
        write_json(e.output_dir(exp)/'evaluation.partial.json',report)
    write_json(e.output_dir(exp)/'evaluation.json',report)
    write_json(e.output_dir(exp)/'evaluated.json',dict(design_signature=e.design_signature(),results=str(e.output_dir(exp)/'evaluation.json')))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['eval']);p.add_argument('--filter',required=True);a=p.parse_args()
    if e.SUITE!='mtp-unified':raise RuntimeError('Select mtp-unified suite')
    matches=e.collect_experiments(a.filter)
    if len(matches)!=1:raise ValueError('Select one unified model')
    def expired(*args):raise TimeoutError('Unified evaluation time cap')
    signal.signal(signal.SIGALRM,expired);signal.alarm(e.UNIFIED_TIMEOUT)
    try:evaluate(matches[0])
    finally:signal.alarm(0)


if __name__=='__main__':main()
