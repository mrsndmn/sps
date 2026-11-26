"""FP16-backbone control for cached verification discrepancies with FP32 head.

Same FP32 source export and previous prefix/output cases. Both weight rounding
and backbone arithmetic change when selecting FP16 instead of BF16; this is not
an arithmetic-only intervention. FP16 has a smaller exponent range; nonfinite
distributions fail validation. Original failed-gate evidence stays immutable.
"""
import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shlex
import time
from unittest.mock import patch

from scripts import sampling_precision as precision
from scripts import sampling_cache_trace as cache
from scripts import sampling_audit as audit
from scripts import sampling_trace as trace
from scripts import mlspace_launch as launch

ROOT = precision.ROOT.parent / 'sps-sampling-backbone-precision'
SOURCE = precision.root()
BASE_IDENTITY = precision.identity

@dataclass(frozen=True)
class Experiment(precision.Experiment):
    dtype: str = 'float16'

experiment_configs = [Experiment('fp16-four', 'sps_four_slots', True),
                      Experiment('fp16-mtp', 'sps_mtp', True)]

@dataclass(frozen=True)
class Job:
    name: str = 'fp16-backbone'
    num_gpus: int = 1
    hours: int = 2

job_configs = [Job()]


def root():
    return ROOT / 'runs/out' / job_configs[0].name


def identity():
    prior = json.loads((SOURCE / 'identity.json').read_text())
    if prior != BASE_IDENTITY():
        raise RuntimeError('Precision baseline inputs changed')
    files = [SOURCE / 'report.json', cache.root() / 'report.json', cache.root() / 'cases.json']
    for name, arm in [('fp32head-four', 'sps'), ('fp32head-mtp', 'sps_mtp')]:
        files.append(SOURCE / name / 'decode' / f'{arm}-eager.json')
        dist_arm = 'sps_four_slots' if arm == 'sps' else arm
        files += [SOURCE / name / 'distributions' / f'{dist_arm}.{s}' for s in ('json', 'npz')]
    return dict(audit=prior['audit'], precision=prior, code=audit.common.file_hash(Path(__file__)),
                evidence={str(p): audit.common.file_hash(p) for p in files})


class BackbonePrecisionWorkerExtension(precision.PrecisionWorkerExtension):
    def sps_backbone_precision_info(self):
        import torch
        model = self.model_runner.model
        parameters = list(model.parameters())
        return dict(parameter_dtypes=sorted({str(p.dtype) for p in parameters if p.is_floating_point()}),
                    finite=all(bool(torch.isfinite(p).all().item()) for p in parameters),
                    **self.sps_precision_info())


def engine_worker(exp, task):
    import vllm
    original = vllm.LLM
    observed = []
    class BackboneLLM(original):
        def __init__(self, *args, **kwargs):
            # precision.engine_worker sets the head and trace extension first;
            # this outer intervention substitutes dtype and an inherited extension.
            kwargs.update(dtype=exp.dtype,
                          worker_extension_cls='scripts.sampling_backbone_precision.BackbonePrecisionWorkerExtension')
            super().__init__(*args, **kwargs)
            observed.extend(self.collective_rpc('sps_backbone_precision_info'))
            if len(observed) != 1 or observed[0]['parameter_dtypes'] != ['torch.float16'] or not observed[0]['finite']:
                raise RuntimeError('FP16 weights absent or nonfinite after conversion')
            if observed[0]['head_dtype'] != 'torch.float32':
                raise RuntimeError('Expected unchanged FP32 head')
    with patch.object(vllm, 'LLM', BackboneLLM), patch.object(precision, 'root', root):
        precision.engine_worker(exp, task)
    launch.store_json(root() / exp.name / task / 'backbone.json', dict(experiment=exp.__dict__, observed=observed))


def validate_decodes(group):
    for row in group["samples"]:
        values = row["sampled_logprobs"]
        if len(values) != len(row["tokens"]) or not all(math.isfinite(v) for v in values):
            raise RuntimeError("Missing or nonfinite sampled log probabilities")


def report():
    cases = json.loads((precision.TRACE_SOURCE / 'cases.json').read_text())
    matrices = {}
    for exp in experiment_configs:
        matrices[exp.name] = precision.load_matrix(root() / exp.name / 'distributions', exp.arm, cases)
    for name, arm in [('fp32head-four', 'sps_four_slots'), ('fp32head-mtp', 'sps_mtp')]:
        matrices[name] = precision.load_matrix(SOURCE / name / 'distributions', arm, cases)
    comparisons = [dict(case=c['name'], a=a, b=b, **trace.compare(matrices[a][i], matrices[b][i]))
                   for i, c in enumerate(cases) for a, b in
                   [('fp16-four', 'fp16-mtp'), ('fp32head-four', 'fp16-four'), ('fp32head-mtp', 'fp16-mtp')]]
    decodes = {exp.name: json.loads((root() / exp.name / 'decode' /
                f"{'sps' if exp.arm == 'sps_four_slots' else exp.arm}-eager.json").read_text())
               for exp in experiment_configs}
    for group in decodes.values():
        validate_decodes(group)
    cached = audit.compare_decodes({'sps-eager': decodes['fp16-four'], 'mtp': decodes['fp16-mtp']})
    result = dict(identity=json.loads((root() / 'identity.json').read_text()), comparisons=comparisons,
                  cached=cached, passed=True, diagnostic_completed=True,
                  scope='FP16-backbone numerical control with FP32 head; weight rounding and arithmetic both change from the same FP32 source export. Diagnostic completion is not a pass of the original BF16 sampling gate. Finite-weight and full-distribution checks do not prove all cached activations finite.')
    launch.store_json(root() / 'report.json', result)
    launch.store_json(root() / 'evaluated.json', result)


def worker():
    current = identity()
    path = root() / 'identity.json'
    if path.exists() and json.loads(path.read_text()) != current:
        raise RuntimeError('Backbone diagnostic inputs changed')
    launch.store_json(path, current)
    env = audit.runtime.runtime_environment()
    env.update(SPS_VLLM_SOURCE=str(audit.graphs.SITE), VLLM_USE_V2_MODEL_RUNNER='0')
    env['PYTHONPATH'] = str(audit.graphs.SITE) + ':' + env.get('PYTHONPATH', str(Path.cwd()) + ':' + str(Path.cwd() / 'src'))
    deadline = time.monotonic() + job_configs[0].hours * 3600 - 300
    for exp in experiment_configs:
        for task in ('distributions', 'decode'):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError('Backbone precision deadline exceeded')
            audit.bench.child([audit.runtime.RUNTIME / 'bin/python', '-m', 'scripts.sampling_backbone_precision',
                               '--engine-worker', exp.name, '--task', task], env, remaining / 3600)
    report()


def payload(exp, staged):
    body = audit.bench.payload(audit.source.Experiment(exp.name, 'audit'), staged, 'audit')
    body['job_desc'] = f'sps-sampling-backbone-precision {exp.name} #d.tarasov @mrsndmn'
    body['script'] = f'cd {shlex.quote(str(staged))} && ' + shlex.join([str(launch.PYTHON), '-m', 'scripts.sampling_backbone_precision', '--worker'])
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--engine-worker', choices=[e.name for e in experiment_configs])
    parser.add_argument('--task', choices=('distributions', 'decode'))
    args = parser.parse_args()
    if args.engine_worker:
        if not args.task:
            parser.error('--engine-worker requires --task')
        return engine_worker(next(e for e in experiment_configs if e.name == args.engine_worker), args.task)
    if args.worker:
        return worker()
    with patch.object(audit, 'ROOT', ROOT), patch.object(audit, 'experiment_configs', job_configs), \
         patch.object(audit, 'input_identity', identity), patch.object(audit, 'output_dir', root), \
         patch.object(audit, 'payload', payload), patch.object(launch, 'ARTIFACTS_ROOT', ROOT), patch.object(launch, 'RUN_ROOT', ROOT / 'runs'):
        if args.dry:
            return audit.submit(True)
        with launch.submission_lock():
            return audit.submit()


if __name__ == '__main__':
    main()
