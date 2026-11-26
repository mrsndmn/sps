"""Matched-slot cached decoding and FP32-head diagnostic after BF16 trace.

Preserves both previous experiments. Completion means diagnostics collected, not
that the original sampling gate passed. No serving speed claims are made here.
"""
from orchestration.named.publication import work_path
from experiments.launch import run_filter
import argparse
import json
import os
from pathlib import Path
import shlex
import time
from unittest.mock import patch
import numpy as np
from workloads.serving import sampling_audit as audit
from workloads.serving import sampling_trace as trace
from orchestration import mlspace_launch as launch

from experiments.sampling_precision import (
    Experiment,
    experiment_configs,
    Job,
    job_configs,
)

ROOT=audit.ROOT.parent/'sps-sampling-precision'
AUDIT_SOURCE=audit.output_dir()
TRACE_SOURCE=trace.output_dir()


def root(job=job_configs[0]):return work_path(ROOT/'runs/out'/job.name)


def identity():
    current=audit.recorded_input_identity(AUDIT_SOURCE)
    prior=json.loads((TRACE_SOURCE/'identity.json').read_text())
    if prior['audit']!=current:raise RuntimeError('Trace used another model/runtime')
    for name,expected in prior['evidence'].items():
        if audit.common.file_hash(AUDIT_SOURCE/name)!=expected:raise RuntimeError('Original audit evidence changed')
    files=[TRACE_SOURCE/'identity.json',TRACE_SOURCE/'cases.json']
    files += [TRACE_SOURCE/f'{arm}.{suffix}' for arm in (*trace.ARMS,*trace.NATIVE_ARMS) for suffix in ('npz','json')]
    files += [AUDIT_SOURCE/'prefixes.json',AUDIT_SOURCE/'sps-eager.json',AUDIT_SOURCE/'sps_mtp-eager.json']
    return dict(audit=current,
                evidence={str(p):audit.common.file_hash(p) for p in files})


def recorded_identity(job=job_configs[0]):
    recorded = json.loads((root(job)/'identity.json').read_text())
    report = json.loads((root(job)/'report.json').read_text())
    current = identity()
    # Revalidate the evidence without requiring the historical producer code.
    if report['identity'] != recorded or any(recorded.get(key) != value for key, value in current.items()):
        raise RuntimeError('Recorded precision inputs changed')
    return recorded


class PrecisionWorkerExtension(trace.TraceWorkerExtension):
    def sps_precision_info(self):
        model=self.model_runner.model
        return dict(head_dtype=str(model.logits_processor.head_dtype),horizons=model.sps_horizons)


def engine_worker(exp,task, job=job_configs[0]):
    import vllm
    from workloads.serving import vllm_sps_evaluate as evaluator
    original=vllm.LLM
    destination=root(job)/exp.name/task
    launch.store_json(destination/'identity.json',
        json.loads((root(job)/'identity.json').read_text())['audit'] if task=='decode' else
        dict(audit=json.loads((root(job)/'identity.json').read_text())['audit']))
    launch.store_json(destination/'prefixes.json',json.loads((AUDIT_SOURCE/'prefixes.json').read_text()))
    launch.store_json(destination/'cases.json',json.loads((TRACE_SOURCE/'cases.json').read_text()))
    observed=[]
    class ConfiguredLLM(original):
        def __init__(self,*args,**kwargs):
            if exp.fp32_head:kwargs['hf_overrides']={'head_dtype':'float32'}
            kwargs['worker_extension_cls']='workloads.serving.sampling_precision.PrecisionWorkerExtension'
            super().__init__(*args,**kwargs)
            if task=='decode' and exp.arm=='sps_four_slots':
                self.collective_rpc('sps_trace_four_slots')
        def generate(self,*args,**kwargs):
            if not observed:
                observed.extend(self.collective_rpc('sps_precision_info'))
                expected_horizons=1 if exp.arm=='sps' else 4
                if len(observed)!=1 or observed[0]['horizons']!=expected_horizons:
                    raise RuntimeError('Prediction slot intervention failed')
                if exp.fp32_head and observed[0]['head_dtype']!='torch.float32':
                    raise RuntimeError('FP32 head override did not take effect')
                if not exp.fp32_head and observed[0]['head_dtype'] not in ('None','torch.bfloat16'):
                    raise RuntimeError('Unexpected baseline head precision')
            return super().generate(*args,**kwargs)
    with patch.object(vllm,'LLM',ConfiguredLLM):
        if task=='distributions':
            with patch.object(trace,'output_dir',lambda job: destination):trace.engine_worker(exp.arm)
        else:
            with patch.object(audit,'output_dir',lambda job: destination):
                audit.decode_worker('sps' if exp.arm=='sps_four_slots' else exp.arm,'eager')
    launch.store_json(destination/'precision.json',dict(experiment=exp.__dict__,observed=observed))


def load_matrix(directory,arm,cases):
    meta=json.loads((directory/f'{arm}.json').read_text())
    if meta['cases'] != [x['name'] for x in cases] or meta['sha256']!=audit.common.file_hash(directory/f'{arm}.npz'):
        raise RuntimeError('Mismatched distribution evidence')
    matrix=np.load(directory/f'{arm}.npz')['logprobs']
    if len(matrix)!=len(cases):raise RuntimeError('Missing distribution cases')
    return matrix


def report(job=job_configs[0]):
    cases=json.loads((TRACE_SOURCE/'cases.json').read_text())
    matrices={arm:load_matrix(TRACE_SOURCE,arm,cases) for arm in (*trace.ARMS,*trace.NATIVE_ARMS)}
    # Reanalyse saved arrays with stable ties; do not rewrite original trace artifacts.
    pairs=[('sps','sps_mtp'),('sps_four_slots','sps_mtp'),
           ('native-fp32-one','native-fp32-four'),('native-bf16-one','native-bf16-four')]
    for exp in experiment_configs:
        matrices[exp.name]=load_matrix(root(job)/exp.name/'distributions',exp.arm,cases)
    pairs += [('fp32head-one','fp32head-four'),('fp32head-four','fp32head-mtp'),('sps_four_slots','bf16-four')]
    comparisons=[dict(case=case['name'],a=a,b=b,**trace.compare(matrices[a][i],matrices[b][i]))
                 for i,case in enumerate(cases) for a,b in pairs]
    cached={}
    for exp in experiment_configs:
        arm='sps' if exp.arm=='sps_four_slots' else exp.arm
        cached[exp.name]=json.loads((root(job)/exp.name/'decode'/f'{arm}-eager.json').read_text())
    original_mtp=json.loads((AUDIT_SOURCE/'sps_mtp-eager.json').read_text())
    cache_results={
        'bf16_four_vs_mtp':audit.compare_decodes({'sps-eager':cached['bf16-four'],'mtp':original_mtp}),
        'fp32head_four_vs_mtp':audit.compare_decodes({'sps-eager':cached['fp32head-four'],'mtp':cached['fp32head-mtp']}),
        'fp32head_one_vs_four':audit.compare_decodes({'sps-eager':cached['fp32head-one'],'four':cached['fp32head-four']})}
    result=dict(identity=json.loads((root(job)/'identity.json').read_text()),passed=True,diagnostic_completed=True,
        comparisons=comparisons,cached=cache_results,
        scope='Diagnostic completion only; original sampling gate unchanged. Eager execution, no performance claim. Stable first-index argmax and tie counts replace unstable sorting in this derived analysis.')
    launch.store_json(root(job)/'report.json',result);launch.store_json(root(job)/'evaluated.json',result)


def worker(job=job_configs[0]):
    current=identity();path=root(job)/'identity.json'
    if path.exists() and json.loads(path.read_text())!=current:raise RuntimeError('Precision inputs changed')
    launch.store_json(path,current)
    env=audit.runtime.runtime_environment()
    env.update(SPS_VLLM_SOURCE=str(audit.graphs.SITE),VLLM_USE_V2_MODEL_RUNNER='0')
    env['PYTHONPATH']=str(audit.graphs.SITE)+':'+env.get('PYTHONPATH',str(Path.cwd())+':'+str(Path.cwd()/'src'))
    deadline=time.monotonic()+job.hours*3600-300
    for exp in experiment_configs:
        for task in ('distributions','decode'):
            remaining=deadline-time.monotonic()
            if remaining<=0:raise RuntimeError('Precision diagnostic deadline exceeded')
            audit.bench.child([audit.runtime.RUNTIME/'bin/python','scripts/serving/sampling_precision.py', '--filter', job.name,
                               '--engine-worker',exp.name,'--task',task],env,remaining/3600)
    report(job=job)


def payload(exp,staged):
    body=audit.bench.payload(audit.source.Experiment(exp.name,'audit'),staged,'audit')
    body['job_desc']=f'sps-sampling-precision {exp.name} #d.tarasov @mrsndmn'
    body['script']=f'cd {shlex.quote(str(staged))} && '+shlex.join([str(launch.PYTHON),'scripts/serving/sampling_precision.py', '--worker', '--filter', exp.name])
    return body


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry',action='store_true');parser.add_argument('--worker',action='store_true')
    parser.add_argument('--engine-worker',choices=[x.name for x in experiment_configs])
    parser.add_argument('--task',choices=('distributions','decode'))
    jobs = run_filter(parser, job_configs)
    args = parser.parse_args()
    job = jobs[args.filter]
    if args.engine_worker:
        if not args.task:parser.error('--engine-worker requires --task')
        return engine_worker(next(x for x in experiment_configs if x.name==args.engine_worker),args.task, job=job)
    if args.worker:return worker(job=job)
    with patch.object(launch, "ARTIFACTS_ROOT", ROOT), patch.object(launch, "RUN_ROOT", ROOT / "runs"):
        arguments = (job, ROOT, root(job), identity, payload)
        if args.dry:
            return audit.submit(*arguments, dry=True)
        with launch.submission_lock():
            return audit.submit(*arguments)
