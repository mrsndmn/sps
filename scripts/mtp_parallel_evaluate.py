"""Evaluation-only parallel verification of existing parent/full SPS MTP drafts."""
import argparse
import json
import signal

import torch
import experiments as e
from scripts.mtp_common import (load_mtp,load_reference,TokenSequences,
    validation_starts,validate_evaluation_inputs,warmstart_checkpoint,file_sha256)
from scripts.mtp_evaluate import (benchmark_reference,benchmark_speculative,
    NativePrefixVerifier,probabilities)
from scripts.pilot_run import write_json
from modeling.models.sps.parallel_verify import ParallelPrefixVerifier


@torch.inference_mode()
def audit(reference,prompts):
    """Independent native comparisons, including EOS and rejected-branch recovery."""
    maximum=0.;logit_error=0.;argmax_mismatches=0;rows=0
    for prefix in prompts[:e.MTP_PARALLEL_AUDIT_PROMPTS]:
        parallel=ParallelPrefixVerifier(reference,e.MTP_NEW_TOKENS)
        native=NativePrefixVerifier(reference,e.MTP_NEW_TOKENS)
        # Include a document reset inside a block, followed by rejection before it.
        draft=torch.tensor([262,reference.config.eos_token_id,198,257],device=prefix.device)
        for current in (prefix,torch.cat((prefix,draft[:1],draft.new_tensor([13])))):
            p=native.verify_logits(current,draft)
            q=parallel.verify_logits(current,draft)
            maximum=max(maximum,float((probabilities(p,reference.config)-probabilities(q,reference.config)).abs().max()))
            logit_error=max(logit_error,float((p-q).abs().max()))
            argmax_mismatches+=int((p.argmax(-1)!=q.argmax(-1)).sum())
            rows+=p.shape[0]
    if maximum > e.MTP_PARALLEL_PROB_ATOL:
        raise RuntimeError(f'Parallel/native probability drift {maximum} exceeds audit tolerance')
    return dict(max_probability_error=maximum,max_logit_error=logit_error,
        audited_rows=rows,argmax_mismatches=argmax_mismatches,
        probability_atol=e.MTP_PARALLEL_PROB_ATOL,
        bitwise_native_equivalent=logit_error==0.)


def aggregate(rows,baseline,n):
    total=sum(r['total_seconds'] for r in rows)
    base_total=sum(r['total_seconds'] for r in baseline)
    throughput=sum(r['tokens'] for r in rows)/total
    base_throughput=sum(r['tokens'] for r in baseline)/base_total
    accepted=[sum(r['accepted_by_horizon'][j] for r in rows) for j in range(n)]
    proposed=[sum(r['proposed_by_horizon'][j] for r in rows) for j in range(n)]
    return dict(prompts=rows,tokens_per_second=throughput,baseline_tokens_per_second=base_throughput,
        speedup=throughput/base_throughput,total_seconds=total,baseline_total_seconds=base_total,
        accepted_by_horizon=accepted,proposed_by_horizon=proposed,
        acceptance_by_horizon=[a/p if p else None for a,p in zip(accepted,proposed)],
        accepted_tokens_per_round=sum(accepted)/sum(r['rounds'] for r in rows))


@torch.inference_mode()
def evaluate(exp):
    if not torch.cuda.is_available(): raise RuntimeError('Production timing requires CUDA')
    trained=json.loads(e.training_marker(exp).read_text())
    validate_evaluation_inputs(trained)
    checkpoint=trained['checkpoint']
    state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if state.get('design_signature')!=trained['design_signature'] or state.get('horizons')!=exp.horizons:
        raise RuntimeError('Completed checkpoint identity mismatch')
    del state
    reference=load_reference('cuda').eval().requires_grad_(False)
    data=TokenSequences('val',e.MTP_PROMPT_TOKENS,exp.horizons)
    starts=validation_starts(len(data.tokens),e.MTP_PROMPT_TOKENS,e.MTP_EVAL_PROMPTS)
    prompts=[data[i][:e.MTP_PROMPT_TOKENS].cuda() for i in starts]
    report=dict(verifier_backend='parallel_q2',checkpoint=checkpoint,
        checkpoint_sha256=file_sha256(checkpoint),source_sha256=trained['source_sha256'],
        data_manifest_sha256=trained['data_manifest_sha256'],horizons=exp.horizons,
        settings=dict(prompts=len(prompts),prompt_tokens=e.MTP_PROMPT_TOKENS,
            max_new_tokens=e.MTP_NEW_TOKENS,temperature=e.MTP_TEMPERATURE,seed=e.MTP_SEED,batch_size=1),
        limitations=['Exact acceptance/residual correction relative to computed block probabilities; floating point block GEMMs can differ from native single-token decoding. Native sampling distribution is not claimed bitwise identical.',
            'One seed/workload; timings include proposal, prefill, verification, cache reconciliation and probability/acceptance logic.'],
        numerical_audit=audit(reference,prompts),baseline={})
    for mode in ('greedy','sample'):
        benchmark_reference(reference,prompts[0],mode=mode)
        report['baseline'][mode]=[benchmark_reference(reference,p,mode=mode,seed=e.MTP_SEED+i) for i,p in enumerate(prompts)]
    all_equal=True
    for phase,path in [('before',warmstart_checkpoint(exp.horizons)),('after',checkpoint)]:
        model=load_mtp(exp.horizons,'cuda',path,backend=e.MTP_INFERENCE_BACKEND).eval()
        report[phase]=dict(checkpoint=str(path),checkpoint_sha256=file_sha256(path),generation={})
        for mode in ('greedy','sample'):
            benchmark_speculative(model,prompts[0],mode=mode,reference=reference,verifier_factory=ParallelPrefixVerifier)
            rows=[benchmark_speculative(model,p,mode=mode,seed=e.MTP_SEED+i,reference=reference,
                verifier_factory=ParallelPrefixVerifier) for i,p in enumerate(prompts)]
            result=aggregate(rows,report['baseline'][mode],exp.horizons)
            if mode=='greedy':
                matches=[r['token_ids']==b['token_ids'] for r,b in zip(rows,report['baseline'][mode])]
                result['native_exact_prompts']=sum(matches)
                result['native_total_prompts']=len(matches)
                result['native_greedy_all_equal']=all(matches)
                all_equal &= all(matches)
            report[phase]['generation'][mode]=result
        del model
        torch.cuda.empty_cache()
    report['native_greedy_all_equal']=all_equal
    dest=e.output_dir(exp)
    write_json(dest/'evaluation.json',report)
    # Numeric mismatches are evidence, never silently substitute sequential timing.
    write_json(dest/'evaluated.json',dict(design_signature=e.design_signature(),
        results=str(dest/'evaluation.json'),native_greedy_all_equal=all_equal))
    print(json.dumps(dict(name=exp.name,native_greedy_all_equal=all_equal,
        results=str(dest/'evaluation.json'))),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['eval']);p.add_argument('--filter',required=True)
    a=p.parse_args()
    if e.SUITE!='mtp-parallel':raise RuntimeError('Select mtp-parallel suite')
    matches=e.collect_experiments(a.filter)
    if len(matches)!=1:raise ValueError('Select one experiment')
    def expired(signum, frame):
        raise TimeoutError('Parallel evaluation exceeded its configured time limit')
    signal.signal(signal.SIGALRM,expired)
    signal.alarm(e.MTP_PARALLEL_TIMEOUT_SECONDS)
    try:
        evaluate(matches[0])
    finally:
        signal.alarm(0)


if __name__=='__main__':main()
