"""Observe target logits before greedy rejection on matched causal histories.

Diagnostic only: FP32 head, four slots, eager synchronous one-request decoding.
Capture divergent histories and compare with cold full-prefix evaluation. The
original failed correctness gate is immutable; instrumentation is not a benchmark.
"""
from orchestration.named.publication import work_path
from experiments.launch import run_filter
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import time
from unittest.mock import patch

import numpy as np
from workloads.serving import sampling_precision as precision
from workloads.serving import sampling_audit as audit
from workloads.serving import sampling_trace as trace
from orchestration import mlspace_launch as launch

from experiments.sampling_cache_trace import (
    Experiment,
    experiment_configs,
    Job,
    job_configs,
)

ROOT = precision.ROOT.parent / 'sps-sampling-cache-trace'
SOURCE = precision.root()
BASE_IDENTITY = precision.recorded_identity


LIMIT = max(audit.OUTPUT_LENGTHS)


def root(job=job_configs[0]):
    return work_path(ROOT / 'runs/out' / job.name)


def digest(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype='<i8').tobytes()).hexdigest()


def identity():
    prior = json.loads((SOURCE / 'identity.json').read_text())
    if prior != BASE_IDENTITY():
        raise RuntimeError('Precision evidence inputs changed')
    files = [SOURCE / 'report.json', SOURCE / 'identity.json',
             SOURCE / 'fp32head-four/decode/sps-eager.json',
             SOURCE / 'fp32head-mtp/decode/sps_mtp-eager.json']
    return dict(precision=prior, evidence={str(p): audit.common.file_hash(p) for p in files})


def evidence():
    prefixes = json.loads((precision.AUDIT_SOURCE / 'prefixes.json').read_text())
    plain = json.loads((SOURCE / 'fp32head-four/decode/sps-eager.json').read_text())['samples']
    spec = json.loads((SOURCE / 'fp32head-mtp/decode/sps_mtp-eager.json').read_text())['samples']
    return prefixes, plain, spec


def select_cases(prefixes, plain, speculative):
    cases = trace.extract_cases(prefixes, plain, speculative)
    return [c for c in cases if 'plain_token' in c]


def row_histories(base, drafts, target_indices, bonus_indices, nrows):
    """Map physical target-logit rows to their causal token histories."""
    indices = [*target_indices, *bonus_indices]
    if len(bonus_indices) != 1 or len(target_indices) != len(drafts) or sorted(indices) != list(range(nrows)):
        raise RuntimeError('Unexpected single-request speculative row layout')
    return [(idx, base + drafts[:j]) for j, idx in enumerate(indices)]


class CacheTraceWorkerExtension(precision.PrecisionWorkerExtension):
    def sps_cache_trace_start(self, cases, destination):
        import torch
        runner = self.model_runner
        if runner.use_async_scheduling or not runner.model.vllm_config.model_config.enforce_eager:
            raise RuntimeError('Trace requires synchronous eager runner')
        if self.sps_precision_info() != dict(head_dtype='torch.float32', horizons=4):
            raise RuntimeError('Expected four slots and FP32 head')
        if hasattr(self, '_cache_trace_original'):
            raise RuntimeError('Observer already installed')
        wanted = {digest(c['tokens']): c for c in cases}
        records = []
        captured = {}
        original = runner._sample
        self._cache_trace_original = original

        def sample(logits, metadata):
            batch = runner.input_batch
            if len(batch.req_ids) != 1:
                raise RuntimeError('Trace supports exactly one request')
            length = int(batch.num_tokens_no_spec[0])
            base = batch.token_ids_cpu[0, :length].tolist()
            if metadata is None:
                drafts, targets, bonuses = [], [], [0]
            else:
                if len(metadata.num_draft_tokens) != 1:
                    raise RuntimeError('Unexpected speculative batch')
                drafts = metadata.draft_token_ids.tolist()
                targets = metadata.target_logits_indices.tolist()
                bonuses = metadata.bonus_logits_indices.tolist()
                if len(drafts) != metadata.num_draft_tokens[0]:
                    raise RuntimeError('Draft metadata inconsistent')
            histories = row_histories(base, drafts, targets, bonuses, len(logits))
            # Clone before sampler can modify logits. Only selected histories retain
            # full distributions; per-step top-token summaries remain small.
            snapshots = []
            for index, history in histories:
                raw = logits[index].detach().float().clone()
                best = int(raw.argmax().item())
                top = torch.topk(raw, 2).values
                key = digest(history)
                rec = dict(history=key, length=len(history), row=index, argmax=best,
                           margin=float((top[0]-top[1]).item()),
                           ties=int((raw == raw[best]).sum().item()))
                if key in wanted:
                    if history != wanted[key]['tokens'] or key in captured:
                        raise RuntimeError('Duplicate or inconsistent captured history')
                    captured[key] = torch.log_softmax(raw, -1).cpu().numpy()
                    rec['case'] = wanted[key]['name']
                snapshots.append(rec)
            output = original(logits, metadata)
            emitted = [int(x) for x in output.sampled_token_ids[0].tolist() if x >= 0]
            if not emitted or len(emitted) > len(histories):
                raise RuntimeError('Invalid greedy sampler output shape')
            # At greedy temperature the emitted accepted/replacement/bonus tokens
            # must equal the corresponding verifier argmax on these exact histories.
            if emitted != [r['argmax'] for r in snapshots[:len(emitted)]]:
                raise RuntimeError('Greedy sampler disagrees with its target logits')
            records.append(dict(base_length=length, base=digest(base), drafts=drafts,
                                emitted=emitted, rows=snapshots))
            return output

        runner._sample = sample
        self._cache_trace_state = (cases, destination, records, captured)
        return self.sps_precision_info()

    def sps_cache_trace_finish(self):
        self.model_runner._sample = self._cache_trace_original
        del self._cache_trace_original
        cases, destination, records, captured = self._cache_trace_state
        del self._cache_trace_state
        missing = [c['name'] for c in cases if digest(c['tokens']) not in captured]
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        launch.store_json(path / 'steps.json', records)
        if missing:
            raise RuntimeError(f'Divergence histories not reproduced: {missing}')
        matrix = np.stack([captured[digest(c['tokens'])] for c in cases])
        for row in matrix:
            trace.check_logprobs(row)
        np.savez_compressed(path / 'logprobs.npz', logprobs=matrix)
        result = dict(cases=[c['name'] for c in cases], steps=len(records),
                      sha256=audit.common.file_hash(path / 'logprobs.npz'))
        launch.store_json(path / 'logprobs.json', result)
        return result


def engine_worker(exp, job=job_configs[0]):
    from vllm import LLM, SamplingParams
    from workloads.serving.vllm_sps_evaluate import check_runtime, engine_arguments
    torch, _ = check_runtime()
    if torch.cuda.device_count() != 1:
        raise RuntimeError('One GPU required')
    saved = json.loads((root(job) / 'identity.json').read_text())
    cases = json.loads((root(job) / 'cases.json').read_text())
    prefixes, plain, speculative = evidence()
    arguments = engine_arguments(audit.source.Experiment(exp.name, exp.arm),
                                saved['precision']['audit']['checkpoint']['export'], audit.source.family_config.EAGLE_REVISION)
    arguments.update(hf_overrides={'head_dtype': 'float32'},
                     worker_extension_cls='workloads.serving.sampling_cache_trace.CacheTraceWorkerExtension')
    llm = LLM(**arguments)
    if exp.arm == 'sps':
        llm.collective_rpc('sps_trace_four_slots')
    destination = root(job) / exp.name
    llm.collective_rpc('sps_cache_trace_start', args=(cases, str(destination)))
    samples = []
    requests = cases if exp.cold else [dict(name=f"prefix-{i}", prefix=i, tokens=prefixes[i]['tokens'])
                                     for i in sorted({c['prefix'] for c in cases})]
    for case in requests:
        out = llm.generate([{'prompt_token_ids': case['tokens']}],
            SamplingParams(temperature=0, max_tokens=1 if exp.cold else LIMIT,
                           ignore_eos=True, seed=audit.SEED, logprobs=1), use_tqdm=False)[0].outputs[0]
        tokens = list(out.token_ids)
        if not exp.cold:
            expected = [r for r in (plain if exp.arm == 'sps' else speculative)
                        if r['prefix'] == case['prefix'] and r['limit'] == LIMIT and r['ignore_eos']]
            if len(expected) != 1 or tokens != expected[0]['tokens'] or out.finish_reason != expected[0]['stop']:
                raise RuntimeError('Instrumented generation did not reproduce precision evidence')
        samples.append(dict(name=case['name'], tokens=tokens, stop=out.finish_reason))
    observed = llm.collective_rpc('sps_cache_trace_finish')
    launch.store_json(destination / 'samples.json', dict(samples=samples, observed=observed, arguments=arguments))


def report(job=job_configs[0]):
    cases = json.loads((root(job) / 'cases.json').read_text())
    matrices = {e.name: precision.load_matrix(root(job) / e.name, 'logprobs', cases) for e in experiment_configs}
    comparisons = [dict(case=c['name'], a=a, b=b, **trace.compare(matrices[a][i], matrices[b][i]))
                   for i, c in enumerate(cases) for a, b in
                   [('sequential', 'speculative'), ('sequential', 'cold'), ('speculative', 'cold')]]
    result = dict(identity=json.loads((root(job) / 'identity.json').read_text()),
                  comparisons=comparisons, passed=True, diagnostic_completed=True,
                  scope='Observer reproduced previous outputs and validated greedy selections against actual target logits. Diagnostic completion does not pass the original sampling gate; cold BF16 backbone is not exact ground truth.')
    launch.store_json(root(job) / 'report.json', result)
    launch.store_json(root(job) / 'evaluated.json', result)


def worker(job=job_configs[0]):
    current = identity()
    path = root(job) / 'identity.json'
    if path.exists() and json.loads(path.read_text()) != current:
        raise RuntimeError('Cache trace inputs changed')
    launch.store_json(path, current)
    cases = select_cases(*evidence())
    if not cases:
        raise RuntimeError('No divergent histories to diagnose')
    launch.store_json(root(job) / 'cases.json', cases)
    env = audit.runtime.runtime_environment()
    env.update(SPS_VLLM_SOURCE=str(audit.graphs.SITE), VLLM_USE_V2_MODEL_RUNNER='0')
    env['PYTHONPATH'] = str(audit.graphs.SITE) + ':' + env.get('PYTHONPATH', str(Path.cwd()) + ':' + str(Path.cwd() / 'src'))
    deadline = time.monotonic() + job.hours * 3600 - 300
    for exp in experiment_configs:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError('Cache trace deadline exceeded')
        audit.bench.child([audit.runtime.RUNTIME / 'bin/python', 'scripts/serving/sampling_cache_trace.py', '--filter', job.name,
                           '--engine-worker', exp.name], env, remaining / 3600)
    report(job=job)


def payload(exp, staged):
    body = audit.bench.payload(audit.source.Experiment(exp.name, 'audit'), staged, 'audit')
    body['job_desc'] = f'sps-sampling-cache-trace {exp.name} #d.tarasov @mrsndmn'
    body['script'] = f'cd {shlex.quote(str(staged))} && ' + shlex.join([str(launch.PYTHON), 'scripts/serving/sampling_cache_trace.py', '--worker', '--filter', exp.name])
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--engine-worker', choices=[e.name for e in experiment_configs])
    jobs = run_filter(parser, job_configs)
    args = parser.parse_args()
    job = jobs[args.filter]
    if args.engine_worker:
        return engine_worker(next(e for e in experiment_configs if e.name == args.engine_worker), job=job)
    if args.worker:
        return worker(job=job)
    with patch.object(launch, "ARTIFACTS_ROOT", ROOT), patch.object(launch, "RUN_ROOT", ROOT / "runs"):
        arguments = (job, ROOT, root(job), identity, payload)
        if args.dry:
            return audit.submit(*arguments, dry=True)
        with launch.submission_lock():
            return audit.submit(*arguments)
