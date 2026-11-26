"""Diagnose failed greedy parity on identical prefixes without relaxing the gate.

Compare full next-token distributions with one/four prediction slots in native
FP32/BF16 and vLLM BF16. The four-slot non-speculative arm is a diagnostic eager
intervention, not a benchmark. Existing audit outputs and runtimes stay immutable.
"""
from orchestration.named.publication import work_path
import argparse
import json
import os
from pathlib import Path
import shlex
import time
from unittest.mock import patch

import numpy as np

from workloads.serving import sampling_audit as audit
from orchestration import mlspace_launch as launch

from experiments.sampling_trace import family_config as sampling_trace_settings
from experiments.sampling_trace import Experiment, experiment_configs
ARMS = sampling_trace_settings.ARMS
NATIVE_ARMS = sampling_trace_settings.NATIVE_ARMS

ROOT = audit.ROOT.parent/'sps-sampling-trace'
SOURCE = audit.output_dir()
FILES = ('identity.json','prefixes.json','kernel.json','report.json',
         'sps-eager.json','sps_mtp-eager.json','sps-graphs.json','sps_mtp-graphs.json')


def output_dir():
    return work_path(ROOT/'runs/out'/experiment_configs[0].name)


def identity():
    return dict(audit=audit.recorded_input_identity(SOURCE),
                evidence={name:audit.common.file_hash(SOURCE/name) for name in FILES})


def extract_cases(prefixes, plain, speculative):
    """Fixed original prefixes plus unique first-divergence histories, paired exactly."""
    cases = [dict(name=f'initial-{i}',prefix=i,offset=0,tokens=x['tokens'],kind='initial')
             for i,x in enumerate(prefixes)]
    a = {(r['prefix'],r['limit'],r['ignore_eos']):r for r in plain}
    b = {(r['prefix'],r['limit'],r['ignore_eos']):r for r in speculative}
    if len(a)!=len(plain) or len(b)!=len(speculative) or set(a)!=set(b):
        raise ValueError('Unpaired or duplicate audit evidence')
    for i in range(len(prefixes)):
        key=(i,max(audit.OUTPUT_LENGTHS),True)
        x,y=a[key]['tokens'],b[key]['tokens']
        offset=next((j for j,(t,u) in enumerate(zip(x,y)) if t!=u),None)
        if offset is None:
            continue
        if offset == 0:
            cases[i].update(plain_token=x[0],speculative_token=y[0])
        else:
            cases.append(dict(name=f'divergence-{i}-{offset}',prefix=i,offset=offset,
                tokens=prefixes[i]['tokens']+x[:offset],kind='divergence',
                plain_token=x[offset],speculative_token=y[offset]))
    return cases


class TraceWorkerExtension:
    def sps_trace_four_slots(self):
        """Change eager model width only; no proposer or scheduler intervention."""
        import torch
        model=self.model_runner.model
        config=self.model_runner.vllm_config
        if not config.model_config.enforce_eager or config.speculative_config is not None:
            raise RuntimeError('Four-slot intervention requires eager non-speculative engine')
        if model.sps_horizons != 1 or model.config.sps_horizons != 4:
            raise RuntimeError('Unexpected prediction horizons')
        model.sps_horizons=4
        model.sps_hidden=torch.empty((model.sps_hidden.shape[0],4,model.config.hidden_size),
                                    device=model.sps_hidden.device,dtype=model.sps_hidden.dtype)
        return dict(horizons=model.sps_horizons,speculation=False)


def check_logprobs(values):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=1 or np.isnan(values).any() or np.isposinf(values).any() or not np.isfinite(values).any():
        raise ValueError('Invalid log probabilities')
    if abs(np.exp(values).sum()-1) > 2e-5:
        raise ValueError('Log probabilities do not cover the full normalized vocabulary')
    return values


def save_distributions(name, cases, arrays, metadata):
    if len(arrays)!=len(cases):
        raise ValueError('Missing distributions')
    matrix=np.stack([check_logprobs(x) for x in arrays])
    root=output_dir();root.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(root/f'{name}.npz',logprobs=matrix)
    launch.store_json(root/f'{name}.json',dict(cases=[x['name'] for x in cases],metadata=metadata,
                                             sha256=audit.common.file_hash(root/f'{name}.npz')))


def engine_worker(arm):
    from vllm import LLM,SamplingParams
    from workloads.serving.vllm_sps_evaluate import check_runtime,engine_arguments
    torch,_=check_runtime()
    if torch.cuda.device_count()!=1:raise RuntimeError('One GPU required')
    saved=json.loads((output_dir()/'identity.json').read_text())
    cases=json.loads((output_dir()/'cases.json').read_text())
    arguments=engine_arguments(audit.source.Experiment(arm,'sps' if arm=='sps_four_slots' else arm),
                               saved['audit']['checkpoint']['export'],audit.source.family_config.EAGLE_REVISION)
    arguments.update(max_logprobs=-1,logprobs_mode='raw_logprobs',
                     worker_extension_cls='workloads.serving.sampling_trace.TraceWorkerExtension')
    llm=LLM(**arguments)
    intervention=llm.collective_rpc('sps_trace_four_slots') if arm=='sps_four_slots' else None
    # Some model vocabularies contain padding entries absent from tokenizer.vocab_size.
    config=json.loads((Path(saved['audit']['checkpoint']['export'])/'config.json').read_text())
    vocab=config['vocab_size']
    arrays=[];samples=[]
    for case in cases:
        out=llm.generate([{'prompt_token_ids':case['tokens']}],
            SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,logprobs=-1),use_tqdm=False)[0].outputs[0]
        lp=out.logprobs[0]
        if set(lp)!=set(range(vocab)):
            raise RuntimeError('vLLM did not return every vocabulary probability')
        arrays.append([lp[i].logprob for i in range(vocab)])
        samples.append(dict(case=case['name'],sampled_token=out.token_ids[0],rank=lp[out.token_ids[0]].rank))
    save_distributions(arm,cases,arrays,dict(arguments=arguments,intervention=intervention,samples=samples))


def native_worker():
    import torch
    from modeling.models.sps.pretrained import PretrainedSPS
    if torch.cuda.device_count()!=1:raise RuntimeError('One GPU required')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    saved=json.loads((output_dir()/'identity.json').read_text())
    cases=json.loads((output_dir()/'cases.json').read_text())
    model=audit.common.load_model('cpu',backend='sdpa')
    checkpoint = audit.common.checked_checkpoint(saved['audit']['checkpoint'])
    state=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if not audit.common.matches_training_input(state) or state['step'] != audit.source.family_config.TRAIN_STEPS:
        raise RuntimeError('Checkpoint identity mismatch')
    model.load_state_dict(state['model']);del state
    single=PretrainedSPS(model.backbone,sps=True);single.prediction=model.prediction
    model=model.cuda().eval().requires_grad_(False)
    single=single.cuda().eval().requires_grad_(False)
    # FP32 first preserves original master weights; BF16 conversion follows.
    for label,dtype in (('fp32',torch.float32),('bf16',torch.bfloat16)):
        model.to(dtype);single.to(dtype)
        one,four=[],[]
        with torch.inference_mode():
            for case in cases:
                tokens=torch.tensor([case['tokens']],device='cuda')
                logits=single(tokens).logits[0,-1].float()
                one.append(logits.log_softmax(-1).cpu().numpy())
                del logits
                logits=model(tokens,positions=[tokens.shape[1]-1]).logits[0,0,0].float()
                four.append(logits.log_softmax(-1).cpu().numpy())
                del logits
        for name,arrays in (('one',one),('four',four)):
            save_distributions(f'native-{label}-{name}',cases,arrays,dict(dtype=str(dtype),tf32=False,backend='sdpa'))


def compare(a,b):
    a,b=check_logprobs(a),check_logprobs(b)
    if a.shape!=b.shape:raise ValueError('Vocabulary mismatch')
    p,q=np.exp(a),np.exp(b)
    order_a=np.argsort(-a,kind='stable')[:2];order_b=np.argsort(-b,kind='stable')[:2]
    top=[int(x) for x in set(order_a.tolist()+order_b.tolist())]
    return dict(argmax_a=int(order_a[0]),argmax_b=int(order_b[0]),
        top_ties_a=int(np.sum(a==a[order_a[0]])),top_ties_b=int(np.sum(b==b[order_b[0]])),
        top_gap_a=float(a[order_a[0]]-a[order_a[1]]),top_gap_b=float(b[order_b[0]]-b[order_b[1]]),
        total_variation=float(np.abs(p-q).sum()/2),
        top_candidates={str(i):dict(logprob_a=float(a[i]),logprob_b=float(b[i])) for i in sorted(top)})


def report():
    cases=json.loads((output_dir()/'cases.json').read_text())
    matrices={}
    for arm in (*ARMS,*NATIVE_ARMS):
        meta=json.loads((output_dir()/f'{arm}.json').read_text())
        if meta['cases'] != [x['name'] for x in cases] or meta['sha256']!=audit.common.file_hash(output_dir()/f'{arm}.npz'):
            raise RuntimeError('Unpaired/corrupt trace distributions')
        matrices[arm]=np.load(output_dir()/f'{arm}.npz')['logprobs']
        if len(matrices[arm])!=len(cases):raise RuntimeError('Incomplete trace matrix')
    pairs=(('sps','sps_mtp'),('sps','sps_four_slots'),('sps_four_slots','sps_mtp'),
           ('native-fp32-one','native-fp32-four'),('native-bf16-one','native-bf16-four'),
           ('native-bf16-one','sps'),('native-bf16-four','sps_mtp'))
    comparisons=[dict(case=case['name'],a=a,b=b,**compare(matrices[a][i],matrices[b][i]))
                 for i,case in enumerate(cases) for a,b in pairs]
    value=dict(identity=json.loads((output_dir()/'identity.json').read_text()),
        passed=True,diagnostic_completed=True,comparisons=comparisons,
        scope='Diagnostic completion only, NOT a passed sampling audit. Full-prefix rescoring does not reproduce cached block verification. Original greedy gate remains failed.')
    launch.store_json(output_dir()/'report.json',value)
    launch.store_json(output_dir()/'evaluated.json',value)
    return value


def worker():
    current=identity()
    root=output_dir();path=root/'identity.json'
    if path.exists() and json.loads(path.read_text())!=current:raise RuntimeError('Trace inputs changed')
    launch.store_json(path,current)
    cases=extract_cases(json.loads((SOURCE/'prefixes.json').read_text()),
        json.loads((SOURCE/'sps-eager.json').read_text())['samples'],
        json.loads((SOURCE/'sps_mtp-eager.json').read_text())['samples'])
    launch.store_json(root/'cases.json',cases)
    deadline=time.monotonic()+experiment_configs[0].hours*3600-300
    def child(python,args,env):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise RuntimeError('Trace deadline exceeded')
        audit.bench.child([python,'scripts/serving/sampling_trace.py',*args],env,remaining/3600)
    child(launch.PYTHON,['--native-worker'],os.environ.copy())
    env=audit.runtime.runtime_environment()
    env.update(SPS_VLLM_SOURCE=str(audit.graphs.SITE),VLLM_USE_V2_MODEL_RUNNER='0')
    env['PYTHONPATH']=str(audit.graphs.SITE)+':'+env.get('PYTHONPATH',str(Path.cwd())+':'+str(Path.cwd()/'src'))
    for arm in ARMS:child(audit.runtime.RUNTIME/'bin/python',['--engine-worker',arm],env)
    report()


def payload(exp,staged):
    body=audit.bench.payload(audit.source.Experiment(exp.name,'audit'),staged,'audit')
    body['job_desc']=f'sps-sampling-trace {exp.name} #d.tarasov @mrsndmn'
    body['script']=f'cd {shlex.quote(str(staged))} && '+shlex.join([str(launch.PYTHON),'scripts/serving/sampling_trace.py','--worker'])
    return body


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry',action='store_true');parser.add_argument('--worker',action='store_true')
    parser.add_argument('--native-worker',action='store_true')
    parser.add_argument('--engine-worker',choices=ARMS)
    args=parser.parse_args()
    if args.native_worker:return native_worker()
    if args.engine_worker:return engine_worker(args.engine_worker)
    if args.worker:return worker()
    # Reuse the reviewed bounded/deduplicated launcher, with a distinct identity.
    with patch.object(audit,'ROOT',ROOT),patch.object(audit,'experiment_configs',experiment_configs), \
         patch.object(audit,'input_identity',identity),patch.object(audit,'output_dir',output_dir), \
         patch.object(audit,'payload',payload),patch.object(launch,'ARTIFACTS_ROOT',ROOT), \
         patch.object(launch,'RUN_ROOT',ROOT/'runs'):
        if args.dry:return audit.submit(True)
        with launch.submission_lock():return audit.submit()
