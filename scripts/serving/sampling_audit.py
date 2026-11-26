"""First-wave audit of the pinned SPS speculative runtime, without retraining.

Runs the actual vLLM delta-proposal rejection kernel on known distributions,
then compares extended greedy cached decoding with/without speculation in eager
and graph modes. This is a diagnostic gate, not a proof of full sampling equality.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import time
from unittest.mock import patch

from scripts import mlspace_launch as launch
from scripts import serving_benchmark as bench
from scripts import serving_graphs as graphs
from scripts import serving_runtime as runtime
from scripts import vllm_sps_common as common
from scripts import vllm_sps_config as source

ROOT = source.ROOT.parent / 'sps-sampling-audit'
SAMPLES = 32768
SEED = 9147
PREFIX_LENGTHS = (32, 127, 511, 1023)
OUTPUT_LENGTHS = (1, 2, 3, 4, 5, 32, 128)
KERNEL_CASES = (
    ('uniform', (1,1,1,1,1,1,1,1), 1., 1., 8, (0,1,2)),
    ('skewed', (40,20,15,10,7,4,3,1), 1., 1., 8, (0,4,7)),
    ('thinking', (40,20,15,10,7,4,3,1), .6, .95, 8, (0,1,7)),
    ('nonthinking', (40,20,15,10,7,4,3,1), .7, .8, 8, (0,1,7)),
    ('topk', (40,20,15,10,7,4,3,1), .7, 1., 2, (0,1,7)),
    ('point_mass', (1,0,0,0,0,0,0,0), 1., 1., 8, (0,0,1)),
)

@dataclass(frozen=True)
class Experiment:
    name: str = 'kernel-and-cache'
    num_gpus: int = 1
    hours: int = 4

experiment_configs = [Experiment()]


def signature():
    return hashlib.sha256(json.dumps(dict(code=common.file_hash(Path(__file__)),
        graph_protocol=graphs.protocol_signature()), sort_keys=True).encode()).hexdigest()


def input_identity():
    manifest = graphs.verify_runtime()
    checkpoint = bench.trained()
    if common.file_hash(checkpoint['checkpoint']) != checkpoint['checkpoint_sha256']:
        raise RuntimeError('Checkpoint bytes changed')
    export = Path(checkpoint['export'])
    weights = sorted(export.glob('*.safetensors'))
    if not weights or not (export/'config.json').exists():
        raise RuntimeError('Missing exported model')
    return dict(design_signature=signature(), runtime=manifest, checkpoint=checkpoint,
                export_sha256={p.name:common.file_hash(p) for p in weights+[export/'config.json']},
                rejection_sampler_sha256=common.file_hash(graphs.SITE/'vllm/v1/sample/rejection_sampler.py'))


def output_dir():
    return ROOT/'runs/out'/experiment_configs[0].name


def frequency_check(values, probabilities, trials):
    """Family-wise Hoeffding bound across all planned kernel cells (alpha <= 1e-6)."""
    cells = len(KERNEL_CASES)*4*8
    tolerance = math.sqrt(math.log(2*cells/1e-6)/(2*trials))
    errors = [abs(a/trials-b) for a,b in zip(values, probabilities)]
    return dict(n=trials, counts=values, expected=probabilities, max_error=max(errors),
                tolerance=tolerance, passed=max(errors) <= tolerance and all(n == 0 for n,p in zip(values, probabilities) if p == 0.))


def kernel_worker():
    import torch
    from types import SimpleNamespace
    from vllm.v1.sample.rejection_sampler import rejection_sample, apply_sampling_constraints
    torch.manual_seed(SEED)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one GPU required')
    device = 'cuda'
    results = []
    for name, weights, temperature, top_p, top_k, proposal in KERNEL_CASES:
        width, vocab = len(proposal), len(weights)
        cumulative = torch.arange(1, SAMPLES+1, device=device, dtype=torch.int32)*width
        meta = SimpleNamespace(all_greedy=False, all_random=True, generators={},
            temperature=torch.full((SAMPLES,), temperature, device=device),
            top_p=torch.full((SAMPLES,), top_p, device=device),
            top_k=torch.full((SAMPLES,), top_k, device=device, dtype=torch.int32))
        base = torch.tensor(weights, device=device, dtype=torch.float32).log()
        logits = base.repeat(SAMPLES*width, 1)
        processed = apply_sampling_constraints(logits, cumulative, meta)
        probabilities = processed[0].softmax(-1)
        expected = probabilities.cpu().tolist()
        # An independent CPU implementation checks temperature and support filtering.
        reference = constrained_probabilities(weights, temperature, top_p, top_k)
        if max(abs(a-b) for a,b in zip(reference, expected)) > 1e-6:
            raise RuntimeError(f'Sampling transforms disagree for {name}')
        draft = torch.tensor(proposal, device=device, dtype=torch.int32).repeat(SAMPLES)
        bonus = torch.multinomial(probabilities.repeat(SAMPLES, 1), 1).to(torch.int32)
        actual = rejection_sample(draft, [width]*SAMPLES, width, cumulative, None,
                                  processed, bonus, meta).cpu().tolist()
        positions = []
        for j in range(width+1):
            emitted = [row[j] for row in actual if all(row[k] == proposal[k] for k in range(j))]
            if not emitted:
                positions.append(dict(position=j, n=0, status='unreachable under this proposal'))
                continue
            if any(x < 0 or x >= vocab for x in emitted):
                raise RuntimeError('Invalid emitted token')
            counts = [emitted.count(i) for i in range(vocab)]
            positions.append(dict(position=j, **frequency_check(counts, expected, len(emitted))))
        results.append(dict(case=name, positions=positions))
    # Force each rejection position and all-accepted bonus in the actual greedy kernel.
    forced = []
    for reject_at in range(4):
        draft = torch.tensor([0,1,2], device=device, dtype=torch.int32)
        logits = torch.full((3,8), -100., device=device)
        for j in range(3):
            logits[j, 7 if j==reject_at else j] = 100.
        meta = SimpleNamespace(all_greedy=True, all_random=False, generators={})
        actual = rejection_sample(draft,[3],3,torch.tensor([3],device=device,dtype=torch.int32),
                                  None,logits,torch.tensor([[6]],device=device,dtype=torch.int32),meta)[0].cpu().tolist()
        expected = list(range(reject_at))+([7] if reject_at < 3 else [6])
        emitted = [x for x in actual if x >= 0]
        forced.append(dict(reject_at=reject_at, expected=expected, actual=emitted, passed=emitted==expected))
    report = dict(cases=results, forced=forced,
        passed=all(p.get('passed',True) for r in results for p in r['positions']) and all(x['passed'] for x in forced))
    launch.store_json(output_dir()/'kernel.json', report)
    if not report['passed']:
        raise RuntimeError('Rejection kernel audit failed')


def constrained_probabilities(weights, temperature, top_p, top_k):
    # Same documented convention as vLLM: discard low-probability cumulative
    # mass <= 1-top_p, always keep the maximum; preserve top-k boundary ties.
    logits = [math.log(w)/temperature if w else -math.inf for w in weights]
    cutoff = sorted(logits, reverse=True)[top_k-1]
    values = [math.exp(v-max(logits)) if v >= cutoff else 0. for v in logits]
    total = sum(values)
    probabilities = [v/total for v in values]
    order = sorted(range(len(values)), key=lambda i:logits[i])
    cumulative = 0.
    for i in order[:-1]:
        cumulative += probabilities[i]
        if cumulative <= 1-top_p:
            values[i] = 0.
    total = sum(values)
    return [v/total for v in values]


def prepare_prefixes():
    rows = common.data('val')
    result = []
    for mode in source.GENERATION:
        eligible = sorted((r for r in rows if r['mode']==mode), key=lambda r:len(r['input_ids']), reverse=True)
        if not eligible or len(eligible[0]['input_ids']) < max(PREFIX_LENGTHS):
            raise RuntimeError('No sufficiently long held-out prefix')
        row = eligible[0]
        for length in PREFIX_LENGTHS:
            result.append(dict(mode=mode, length=length, source_id=row['source_id'], tokens=row['input_ids'][:length]))
    return result


def decode_worker(arm, execution):
    import torch
    from vllm import LLM, SamplingParams
    from scripts.vllm_sps_evaluate import check_runtime, engine_arguments
    check_runtime()
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one GPU required')
    identity = json.loads((output_dir()/'identity.json').read_text())
    arguments = engine_arguments(source.Experiment(arm,arm), identity['checkpoint']['export'], source.EAGLE_REVISION)
    if execution == 'graphs':
        arguments = graphs.graph_arguments(lambda: arguments)
    llm = LLM(**arguments)
    if execution == 'graphs':
        llm.collective_rpc('sps_start_graph_tracking')
    results = []
    prefixes = json.loads((output_dir()/'prefixes.json').read_text())
    for i, prefix in enumerate(prefixes):
        for limit in OUTPUT_LENGTHS:
            for ignore_eos in (False, True):
                out = llm.generate([{'prompt_token_ids':prefix['tokens']}],
                    SamplingParams(temperature=0, max_tokens=limit, ignore_eos=ignore_eos,
                                   seed=SEED, logprobs=1),use_tqdm=False)[0].outputs[0]
                tokens = list(out.token_ids)
                results.append(dict(prefix=i, limit=limit, ignore_eos=ignore_eos, tokens=tokens,
                    stop=out.finish_reason, sampled_logprobs=[float(lp[t].logprob) for lp,t in zip(out.logprobs,tokens)]))
    replay = llm.collective_rpc('sps_stop_graph_tracking') if execution=='graphs' else None
    launch.store_json(output_dir()/f'{arm}-{execution}.json', dict(samples=results, replay=replay))
    if replay and not all(x['target_replays'] > 0 for x in replay):
        raise RuntimeError('No graph replay observed')


def compare_decodes(groups):
    baseline = groups['sps-eager']['samples']
    results = {}
    for name, group in groups.items():
        rows = group['samples']
        if len(rows) != len(baseline):
            raise RuntimeError('Incomplete decode audit')
        mismatches = []
        logprob_max = 0.
        for a,b in zip(baseline,rows):
            if any(a[k] != b[k] for k in ('prefix','limit','ignore_eos')):
                raise RuntimeError('Decode audit pairing mismatch')
            if a['tokens'] != b['tokens'] or a['stop'] != b['stop']:
                mismatches.append({k:b[k] for k in ('prefix','limit','ignore_eos')})
            else:
                logprob_max = max(logprob_max, max((abs(x-y) for x,y in zip(a['sampled_logprobs'],b['sampled_logprobs'])),default=0.))
        results[name] = dict(cases=len(rows), mismatches=mismatches, max_sampled_logprob_error_on_matching_sequences=logprob_max)
    return results


def worker():
    identity = input_identity()
    root = output_dir()
    meta = root/'identity.json'
    if meta.exists() and json.loads(meta.read_text()) != identity:
        raise RuntimeError('Existing audit uses other inputs')
    launch.store_json(meta, identity)
    prefixes = prepare_prefixes()
    launch.store_json(root/'prefixes.json', prefixes)
    env = runtime.runtime_environment()
    env.update(SPS_VLLM_SOURCE=str(graphs.SITE), VLLM_USE_V2_MODEL_RUNNER='0')
    env['PYTHONPATH'] = str(graphs.SITE)+':'+env.get('PYTHONPATH', str(Path.cwd())+':'+str(Path.cwd()/'src'))
    deadline = time.monotonic()+experiment_configs[0].hours*3600-300
    def child(*args):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise RuntimeError('Audit deadline exceeded')
        bench.child([runtime.RUNTIME/'bin/python','-m','scripts.sampling_audit',*args], env,remaining/3600)
    child('--kernel-worker')
    groups = {}
    for execution in ('eager','graphs'):
        for arm in ('sps','sps_mtp'):
            child('--decode-worker',arm,'--execution',execution)
            groups[f'{arm}-{execution}'] = json.loads((root/f'{arm}-{execution}.json').read_text())
    comparison = compare_decodes(groups)
    passed = all(not x['mismatches'] for x in comparison.values())
    report = dict(identity=identity,comparison=comparison,passed=passed,
        scope='Known-distribution kernel checks and extended greedy cache parity; not a proof of general sampled sequence equivalence.')
    launch.store_json(root/'report.json',report)
    if not passed:
        raise RuntimeError('Cached greedy parity failed; inspect report.json')
    launch.store_json(root/'evaluated.json',report)


def payload(exp, staged):
    body = bench.payload(source.Experiment(exp.name,'audit'),staged,'audit')
    body['job_desc'] = f'sps-sampling-audit {exp.name} #d.tarasov @mrsndmn'
    body['script'] = f'cd {shlex.quote(str(staged))} && '+shlex.join([str(launch.PYTHON),'-m','scripts.sampling_audit','--worker'])
    return body


def submit(dry=False):
    commit = launch.git('rev-parse','HEAD')
    staged = launch.STAGING_ROOT/commit
    exp = experiment_configs[0]
    if dry:
        print(json.dumps(payload(exp,staged),indent=2)); return
    identity = input_identity()
    marker = output_dir()/'evaluated.json'
    if marker.exists():
        result = json.loads(marker.read_text())
        if result['identity'] != identity or not result['passed']:
            raise RuntimeError('Stale audit completion')
        print('Already completed'); return
    from mls.manager.job.utils import training_job_api_from_profile
    client,options = training_job_api_from_profile('default')
    launch.disable_submission_retries(client)
    active = {}
    for status in ('Pending','Running'):
        response = client.get_list_jobs(options['region'],None,status,1000,0)
        if 'jobs' not in response:
            raise RuntimeError('Cannot establish active jobs')
        active.update({launch.normalize(x['job_desc']):x['job_name'] for x in response['jobs']})
    launch.reconcile_receipts(client,active); launch.reconcile_intents(active)
    body = payload(exp,staged)
    name = launch.normalize(body['job_desc'])
    if name in active:
        job = dict(name=exp.name, job_name=active[name],reused=True)
    else:
        staged = launch.stage_repo(Path(launch.git('rev-parse','--show-toplevel')),commit)
        body = payload(exp,staged); body['region'] = options['region']
        ledger_path=ROOT/'budget.json'
        ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'attempts':[]}
        if sum(x['gpu_hours'] for x in ledger['attempts'])+exp.hours > 3*exp.hours:
            raise RuntimeError('Audit reservation cap exceeded')
        ledger['attempts'].append(dict(gpu_hours=exp.hours,time=time.time()))
        launch.store_json(ledger_path,ledger)
        job=launch.submit_payload(client,body,'eval',exp.name,commit)
    print('__MLS_JOBS_JSON__'); print(json.dumps({'jobs':[job]}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--kernel-worker',action='store_true')
    parser.add_argument('--decode-worker',choices=('sps','sps_mtp'))
    parser.add_argument('--execution',choices=('eager','graphs'),default='eager')
    args=parser.parse_args()
    if args.kernel_worker:return kernel_worker()
    if args.decode_worker:return decode_worker(args.decode_worker,args.execution)
    if args.worker:return worker()
    with patch.object(launch,'ARTIFACTS_ROOT',ROOT),patch.object(launch,'RUN_ROOT',ROOT/'runs'):
        if args.dry:return submit(True)
        with launch.submission_lock():return submit()


if __name__=='__main__':
    main()
