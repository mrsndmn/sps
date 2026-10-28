"""Matched target-first speculative benchmarks for history lookup, heads, AR and MTP."""
import json,time
import torch
import experiments as e
from modeling.models.sps.parallel_verify import ParallelPrefixVerifier
from modeling.mtp_speculative import speculative_generate
from modeling.spec_baselines import lookup_proposal
from scripts.mtp_common import (load_reference,load_mtp,TokenSequences,validation_starts,file_sha256)
from scripts.mtp_evaluate import PrefixCache,probabilities,benchmark_reference
from scripts.mtp_parallel_evaluate import audit
from scripts.spec_baseline_common import checked_state,construct,restore,identities
from scripts.pilot_run import write_json


class Proposal:
    def __init__(self,kind,model,reference,verifier,horizons):
        self.kind=kind;self.model=model;self.reference=reference;self.verifier=verifier;self.horizons=horizons
        self.cache=PrefixCache(model) if kind in {'ar_tiny','existing_mtp'} else None

    def independent(self,prefix):
        if self.kind=='ngram':return lookup_proposal(prefix,self.horizons,
            self.reference.config.vocab_size,max_match=e.BASELINE_LOOKUP_MATCH)[1]
        if self.kind=='medusa':
            # The final token is the target anchor/correction/bonus. Features of
            # its preceding prefix already exist in the verified KV branch.
            features=self.verifier.ensure_features(prefix[:-1])
            return probabilities(self.model(features,self.reference.lm_head)[:self.horizons],self.reference.config)
        return probabilities(self.cache.ensure(prefix)[:self.horizons],self.reference.config)

    def conditional(self,prefix,remaining,mode,generator):
        n=self.horizons;vocab=self.reference.config.vocab_size
        q=torch.zeros(n,vocab,device=prefix.device);q[:,0]=1.
        draft=torch.zeros(n,device=prefix.device,dtype=torch.long);history=prefix
        for j in range(min(n,remaining)):
            q[j]=probabilities(self.cache.ensure(history)[0],self.reference.config)
            draft[j]=(q[j].argmax() if mode=='greedy' else torch.multinomial(q[j],1,generator=generator)[0])
            history=torch.cat((history,draft[j:j+1]))
        return draft,q


@torch.inference_mode()
def benchmark(kind,model,reference,prefix,horizons,mode,seed):
    torch.cuda.synchronize();started=time.perf_counter()
    verifier=ParallelPrefixVerifier(reference,e.BASELINE_NEW_TOKENS)
    first=probabilities(verifier.ensure(prefix),reference.config)
    generator=torch.Generator(device=prefix.device).manual_seed(seed)
    anchor=first.argmax() if mode=='greedy' else torch.multinomial(first,1,generator=generator)[0]
    extended=torch.cat((prefix,anchor[None]))
    proposal=Proposal(kind,model,reference,verifier,horizons);events=[]
    def timed(name,fn,*args):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();result=fn(*args);end.record();events.append((name,start,end));return result
    if int(anchor)==reference.config.eos_token_id or e.BASELINE_NEW_TOKENS==1:
        ids=anchor[None];accepted=proposed=reached=[0]*horizons;rounds=0
    else:
        def verify(history,draft):
            return timed('verification',lambda:probabilities(verifier.verify_logits(history,draft),reference.config))
        result=speculative_generate(extended,
            lambda history:timed('proposal',proposal.independent,history),verify,
            max_new_tokens=e.BASELINE_NEW_TOKENS-1,mode=mode,
            eos_token_id=reference.config.eos_token_id,generator=generator,
            draft_proposal=(lambda *args:timed('proposal',proposal.conditional,*args)) if kind=='ar_tiny' else None)
        ids=torch.cat((anchor[None],result.token_ids));accepted=result.accepted_by_horizon
        proposed=result.proposed_by_horizon;reached=result.reached_by_horizon;rounds=result.rounds
        # Reconcile final state inside timing, like the existing benchmark.
        verifier.ensure(torch.cat((prefix,ids)))
        if proposal.cache is not None:proposal.cache.ensure(torch.cat((prefix,ids)))
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    return dict(token_ids=ids.tolist(),tokens=ids.numel(),initial_target_tokens=1,
        total_seconds=elapsed,rounds=rounds,accepted_by_horizon=list(accepted),
        proposed_by_horizon=list(proposed),reached_by_horizon=list(reached),
        proposal_stream_seconds=sum(s.elapsed_time(t)/1000 for name,s,t in events if name=='proposal'),
        verification_stream_seconds=sum(s.elapsed_time(t)/1000 for name,s,t in events if name=='verification'))


def summarize(rows,baseline,horizons,mode):
    count=lambda key:[sum(row[key][j] for row in rows) for j in range(horizons)]
    accepted,proposed,reached=(count(k) for k in ('accepted_by_horizon','proposed_by_horizon','reached_by_horizon'))
    total=sum(r['total_seconds'] for r in rows);tokens=sum(r['tokens'] for r in rows)
    base_tokens=sum(r['tokens'] for r in baseline);base_seconds=sum(r['total_seconds'] for r in baseline)
    rounds=sum(r['rounds'] for r in rows)
    return dict(prompts=rows,tokens_per_second=tokens/total,baseline_tokens_per_second=base_tokens/base_seconds,
        speedup=(tokens/total)/(base_tokens/base_seconds),accepted_by_horizon=accepted,
        proposed_by_horizon=proposed,reached_by_horizon=reached,
        unconditional_acceptance=[a/p if p else None for a,p in zip(accepted,proposed)],
        conditional_acceptance=[a/r if r else None for a,r in zip(accepted,reached)],
        accepted_draft_tokens_per_round=sum(accepted)/rounds if rounds else None,
        emitted_per_round=(tokens-sum(r['initial_target_tokens'] for r in rows))/rounds if rounds else None,
        proposal_stream_seconds=sum(r['proposal_stream_seconds'] for r in rows),
        verification_stream_seconds=sum(r['verification_stream_seconds'] for r in rows),
        native_exact_prompts=sum(r['token_ids']==b['token_ids'] for r,b in zip(rows,baseline)) if mode=='greedy' else None,
        total_prompts=len(rows),total_seconds=total)


@torch.inference_mode()
def evaluate(exp):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for timing')
    torch.manual_seed(e.MTP_SEED)
    reference=load_reference('cuda').eval().requires_grad_(False)
    trained=None;checkpoint=None
    if exp.trainable or exp.recipe=='existing_mtp':
        trained,state=checked_state(exp);checkpoint=trained['checkpoint']
    if exp.recipe=='medusa':
        from modeling.spec_baselines import MedusaHeads
        model=MedusaHeads(reference.config.hidden_size,e.BASELINE_MEDUSA_HEADS).cuda().eval()
        model.load_state_dict(state['model'])
    elif exp.recipe=='ar_tiny':
        model=construct(exp,'cuda').eval();restore(model,state,exp)
    elif exp.recipe=='existing_mtp':model=load_mtp(exp.horizons,'cuda',checkpoint,backend=e.MTP_INFERENCE_BACKEND).eval()
    else:model=None
    if trained is not None:del state
    data=TokenSequences('val',e.BASELINE_PROMPT_TOKENS,e.BASELINE_RESERVE_HORIZONS)
    starts=validation_starts(len(data.tokens),e.BASELINE_PROMPT_TOKENS,e.BASELINE_EVAL_PROMPTS)
    prompts=[data[i][:e.BASELINE_PROMPT_TOKENS].cuda() for i in starts]
    report=dict(name=exp.name,recipe=exp.recipe,**identities(),checkpoint=checkpoint,
        checkpoint_sha256=file_sha256(checkpoint) if checkpoint else None,
        numerical_audit=audit(reference,prompts),baseline={},variants={},
        settings=dict(prompts=len(prompts),input_tokens=e.BASELINE_PROMPT_TOKENS,
            new_tokens=e.BASELINE_NEW_TOKENS,seed=e.MTP_SEED,temperature=e.MTP_TEMPERATURE,
            anchor_policy='one initial original-target token; heads predict offsets 2..K+1',
            warmup_full_workload_passes=e.BASELINE_WARMUP_PASSES,
            precision='FP32 weights; BF16 target attention; original native generator baseline'),
        limitations=['Block/native floating point differences are audited; native greedy equality is reported, not assumed.',
            'Conditional acceptance divides by actually reached slots. Emitted/round excludes initial target anchor.',
            'Additional-input budget is matched; Medusa inherits pretrained frozen SPS features, AR is trained from scratch.'])
    for mode in ('greedy','sample'):
        for _ in range(e.BASELINE_WARMUP_PASSES):
            for i,p in enumerate(prompts):benchmark_reference(reference,p,mode=mode,seed=e.MTP_SEED+i,
                max_new_tokens=e.BASELINE_NEW_TOKENS)
        report['baseline'][mode]=[benchmark_reference(reference,p,mode=mode,seed=e.MTP_SEED+i,
            max_new_tokens=e.BASELINE_NEW_TOKENS) for i,p in enumerate(prompts)]
    horizons=(exp.horizons,) if exp.recipe=='existing_mtp' else e.BASELINE_EVAL_HORIZONS
    for n in horizons:
        report['variants'][str(n)]={}
        for mode in ('greedy','sample'):
            warm=time.monotonic()
            for _ in range(e.BASELINE_WARMUP_PASSES):
                for i,p in enumerate(prompts):benchmark(exp.recipe,model,reference,p,n,mode,e.MTP_SEED+i)
            warm_seconds=time.monotonic()-warm
            rows=[benchmark(exp.recipe,model,reference,p,n,mode,e.MTP_SEED+i) for i,p in enumerate(prompts)]
            result=summarize(rows,report['baseline'][mode],n,mode);result['warmup_seconds']=warm_seconds
            report['variants'][str(n)][mode]=result
            print(json.dumps(dict(name=exp.name,horizons=n,mode=mode,speedup=result['speedup'],
                tokens_per_second=result['tokens_per_second'],native_exact_prompts=result['native_exact_prompts'])),flush=True)
            write_json(e.output_dir(exp)/'evaluation.partial.json',report)
    write_json(e.output_dir(exp)/'evaluation.json',report)
    write_json(e.output_dir(exp)/'evaluated.json',dict(design_signature=e.design_signature(),
        results=str(e.output_dir(exp)/'evaluation.json')))
