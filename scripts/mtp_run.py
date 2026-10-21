"""MLSpace MTP workers: data reuse, four-GPU probes, bounded adaptation and eval."""
import argparse
import hashlib
import json
import math
import statistics
import time

import experiments as e
from scripts.pilot_run import run_logged, write_json


def file_digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def check_data():
    manifest=json.loads((e.MTP_DATA/'manifest.json').read_text())
    if (manifest['revision']!=e.DATASET_REVISION or manifest['tokenizer']!='gpt2'
            or manifest['repo']!=e.DATASET_REPO or manifest['dtype']!='uint16'):
        raise RuntimeError('Unexpected source data manifest')
    for split in ('train','val'):
        if (e.MTP_DATA/f'{split}.bin').stat().st_size!=2*manifest['tokens'][split]:
            raise RuntimeError('Incomplete prepared corpus')
    from scripts.mtp_common import sequence_starts
    required=e.MTP_TRAIN_STEPS*e.MTP_GLOBAL_BATCH//4
    capacity=[len(sequence_starts(manifest['tokens']['train'],e.MTP_CONTEXT,4,rank)) for rank in range(4)]
    if min(capacity)<required:raise RuntimeError('Insufficient unique rank coverage')
    return dict(manifest_sha256=file_digest(e.MTP_DATA/'manifest.json'),tokens=manifest['tokens'],
                required_sequences_per_rank=required,available_sequences_per_rank=capacity)


def estimate(exp):
    path=e.SANITY_ROOT/'out'/exp.name/'metrics.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    seconds=[r['step_seconds'] for r in rows if r['event']=='train'][-8:]
    if len(seconds)!=8 or min(seconds)<=0:raise RuntimeError('Need eight stable probe steps')
    median=statistics.median(seconds)
    # Reserve warmup/compilation, checkpoint and validation overhead explicitly.
    low=(median*e.MTP_TRAIN_STEPS*1.1+1800)/3600
    high=(max(seconds)*e.MTP_TRAIN_STEPS*1.5+3600)/3600
    return dict(seconds_per_step=median,tokens_per_second=e.MTP_STEP_TOKENS/median,
                projected_training_hours=high,estimated_hours_range=[low,high],
                token_budget=e.TRAIN_TOKENS,probe_steps=len(rows),max_hours=e.MAX_HOURS)


def prepare():
    import torch
    if torch.cuda.device_count()!=4:raise RuntimeError('MTP preparation needs four GPUs')
    from scripts.pilot_headers import prepare as headers
    headers()
    e.SANITY_ROOT.mkdir(parents=True,exist_ok=True)
    data=check_data()
    run_logged([str(e.PYTHON),'-m','pytest','-q','tests/test_mtp_model.py',
                'tests/test_mtp_speculative.py','tests/test_mtp_evaluate.py'],
               e.SANITY_ROOT/'correctness.log',timeout=3600)
    # Verify the actual pretrained source before spending the adaptation budget.
    from scripts.mtp_common import load_mtp, TokenSequences, validation_starts
    from scripts.mtp_evaluate import check_cached_logits, benchmark_reference, benchmark_speculative
    model=load_mtp(e.MTP_MAX_HORIZONS, 'cuda', backend=e.MTP_INFERENCE_BACKEND).eval()
    validation=TokenSequences('val',e.MTP_PROMPT_TOKENS)
    checks=[]
    for start in validation_starts(len(validation.tokens),e.MTP_PROMPT_TOKENS,e.MTP_PRECHECK_PROMPTS):
        prompt=validation[start][:e.MTP_PROMPT_TOKENS].cuda()
        cached=check_cached_logits(model,prompt)
        baseline=benchmark_reference(model.reference,prompt,mode='greedy',max_new_tokens=e.MTP_PRECHECK_NEW_TOKENS)
        speculative=benchmark_speculative(model,prompt,mode='greedy',max_new_tokens=e.MTP_PRECHECK_NEW_TOKENS)
        if baseline['token_ids']!=speculative['token_ids']:
            raise RuntimeError('Production-checkpoint speculative greedy mismatch')
        checks.append(dict(cached=cached,greedy_exact=True))
    del model
    torch.cuda.empty_cache()
    write_json(e.SANITY_ROOT/'source-correctness.json',checks)
    estimates={}
    for exp in e.collect_experiments():
        metrics=e.SANITY_ROOT/'out'/exp.name/'metrics.jsonl'
        if metrics.exists():metrics.unlink()
        run_logged(e.train_command(exp,probe=True),e.SANITY_ROOT/f'{exp.name}.log',timeout=7200)
        estimates[exp.name]=estimate(exp)
        print(json.dumps(dict(name=exp.name,estimate=estimates[exp.name])),flush=True)
    result=dict(design_signature=e.design_signature(),data=data,
                source_sha256=file_digest(e.MTP_SOURCE),estimates=estimates)
    write_json(e.ARTIFACTS_ROOT/'probe-results.json',result)
    if max(x['projected_training_hours'] for x in estimates.values())>e.MAX_HOURS:
        raise RuntimeError('Measured MTP estimate exceeds approved cap; inspect probe-results.json')
    write_json(e.ARTIFACTS_ROOT/'prepared.json',result)


def train(exp):
    prepared=json.loads((e.ARTIFACTS_ROOT/'prepared.json').read_text())
    signature=e.design_signature()
    if prepared['design_signature']!=signature:raise RuntimeError('Stale MTP preparation')
    if prepared['source_sha256']!=file_digest(e.MTP_SOURCE):raise RuntimeError('Source checkpoint changed')
    if prepared['data']['manifest_sha256']!=file_digest(e.MTP_DATA/'manifest.json'):
        raise RuntimeError('Source data changed')
    check_data()
    forecast=prepared['estimates'][exp.name]
    if forecast['token_budget']!=e.TRAIN_TOKENS or forecast['projected_training_hours']>e.MAX_HOURS:
        raise RuntimeError('Unapproved runtime/budget')
    dest=e.output_dir(exp)
    identity=dest/'run-config.json'
    if identity.exists():
        if json.loads(identity.read_text())['design_signature']!=signature:raise RuntimeError('Stale run identity')
    elif list(dest.glob('*.pt')):raise RuntimeError('Missing partial-run identity')
    write_json(identity,dict(design_signature=signature,source_sha256=prepared['source_sha256'],
        command=e.train_command(exp),estimate=forecast))
    start=time.monotonic()
    run_logged(e.train_command(exp),dest/'train.log',timeout=e.MAX_HOURS*3600)
    final=dest/'adaptation_final.pt'
    if not final.is_file():raise RuntimeError('Missing final adaptation checkpoint')
    write_json(dest/'trained.json',dict(checkpoint=str(final),design_signature=signature,
        source_sha256=prepared['source_sha256'],seconds=time.monotonic()-start,estimate=forecast))


def evaluate(exp):
    dest=e.output_dir(exp)
    trained=json.loads((dest/'trained.json').read_text())
    if trained['design_signature']!=e.design_signature():raise RuntimeError('Stale training identity')
    run_logged([str(e.PYTHON),'-m','scripts.mtp_evaluate','--filter',exp.name],
        dest/'eval.log',timeout=8*3600)
    if not (dest/'evaluation.json').is_file():raise RuntimeError('Missing evaluation results')
    write_json(dest/'evaluated.json',dict(design_signature=e.design_signature(),
        results=str(dest/'evaluation.json')))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','train','eval'))
    parser.add_argument('--filter');args=parser.parse_args()
    if e.SUITE!='mtp':raise RuntimeError('Select SPS_EXPERIMENT_SUITE=mtp')
    if args.phase=='prepare':return prepare()
    matches=e.collect_experiments(args.filter)
    if len(matches)!=1:raise ValueError('Select exactly one experiment')
    (train if args.phase=='train' else evaluate)(matches[0])


if __name__=='__main__':main()
