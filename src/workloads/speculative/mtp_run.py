"""MLSpace MTP workers: data reuse, four-GPU probes, bounded adaptation and eval."""
import argparse
import hashlib
import json
import math
import statistics
import shutil
import time

from experiments import mtp
from orchestration import experiment_suite as e
from workloads.pilot.pilot_run import run_logged, write_json


def file_digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def check_data():
    if mtp.family_config.MTP_UNFREEZE and shutil.disk_usage(mtp.family_config.MTP_DATA).free<mtp.family_config.MTP_CHECKPOINT_FREE_BYTES:
        raise RuntimeError('Insufficient free space for full draft checkpoints')
    manifest=json.loads((mtp.family_config.MTP_DATA/'manifest.json').read_text())
    if (manifest['revision']!=e.DATASET_REVISION or manifest['tokenizer']!='gpt2'
            or manifest['repo']!=e.DATASET_REPO or manifest['dtype']!='uint16'):
        raise RuntimeError('Unexpected source data manifest')
    for split in ('train','val'):
        if (mtp.family_config.MTP_DATA/f'{split}.bin').stat().st_size!=2*manifest['tokens'][split]:
            raise RuntimeError('Incomplete prepared corpus')
    from workloads.speculative.mtp_common import sequence_starts
    required=mtp.family_config.MTP_TRAIN_STEPS*mtp.family_config.MTP_GLOBAL_BATCH//4
    parent_samples=mtp.family_config.MTP_PARENT_INPUT_TOKENS//(mtp.family_config.MTP_PARENT_CONTEXT*4) if mtp.family_config.MTP_UNFREEZE else 0
    capacity=[len(sequence_starts(manifest['tokens']['train'],mtp.family_config.MTP_CONTEXT,4,rank,resume=parent_samples)) for rank in range(4)]
    if min(capacity)<required:raise RuntimeError('Insufficient unique rank coverage')
    return dict(manifest_sha256=file_digest(mtp.family_config.MTP_DATA/'manifest.json'),tokens=manifest['tokens'],
                required_sequences_per_rank=required,available_sequences_per_rank=capacity,
                parent_samples_per_rank=parent_samples)


def estimate(exp):
    path=e.SANITY_ROOT/'out'/exp.name/'metrics.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    seconds=[r['step_seconds'] for r in rows if r['event']=='train'][-8:]
    if len(seconds)!=8 or min(seconds)<=0:raise RuntimeError('Need eight stable probe steps')
    median=statistics.median(seconds)
    # Reserve warmup/compilation, checkpoint and validation overhead explicitly.
    low=(median*mtp.family_config.MTP_TRAIN_STEPS*1.1+1800)/3600
    high=(max(seconds)*mtp.family_config.MTP_TRAIN_STEPS*1.5+3600)/3600
    return dict(seconds_per_step=median,tokens_per_second=mtp.family_config.MTP_STEP_TOKENS/median,
                projected_training_hours=high,estimated_hours_range=[low,high],
                token_budget=e.TRAIN_TOKENS,probe_steps=len(rows),max_hours=e.MAX_HOURS)


def prepare():
    import torch
    if torch.cuda.device_count()!=4:raise RuntimeError('MTP preparation needs four GPUs')
    from orchestration.pilot_headers import prepare as headers
    headers()
    e.SANITY_ROOT.mkdir(parents=True,exist_ok=True)
    data=check_data()
    run_logged(['/usr/bin/env','SPS_EXPERIMENT_SUITE=mtp',str(e.PYTHON),'-m','pytest','-q','tests/test_mtp_model.py',
                'tests/test_mtp_speculative.py','tests/test_mtp_evaluate.py',
                'tests/test_mtp_unfreeze_jobs.py','tests/test_mtp_unfreeze_loading.py'],
               e.SANITY_ROOT/'correctness.log',timeout=3600)
    # Verify the actual pretrained source before spending the adaptation budget.
    from workloads.speculative.mtp_common import load_mtp, TokenSequences, validation_starts
    from workloads.speculative.mtp_evaluate import check_cached_logits, benchmark_reference, benchmark_speculative
    model=load_mtp(mtp.family_config.MTP_MAX_HORIZONS, 'cuda', backend=mtp.family_config.MTP_INFERENCE_BACKEND).eval()
    verifier_args={}
    if mtp.family_config.MTP_UNFREEZE:
        from workloads.speculative.mtp_common import load_reference
        verifier_args['reference']=load_reference('cuda').requires_grad_(False).eval()
    validation=TokenSequences('val',mtp.family_config.MTP_PROMPT_TOKENS)
    checks=[]
    for start in validation_starts(len(validation.tokens),mtp.family_config.MTP_PROMPT_TOKENS,mtp.family_config.MTP_PRECHECK_PROMPTS):
        prompt=validation[start][:mtp.family_config.MTP_PROMPT_TOKENS].cuda()
        cached=check_cached_logits(model,prompt,**verifier_args)
        baseline=benchmark_reference(verifier_args.get('reference',model.reference),prompt,mode='greedy',max_new_tokens=mtp.family_config.MTP_PRECHECK_NEW_TOKENS)
        speculative=benchmark_speculative(model,prompt,mode='greedy',max_new_tokens=mtp.family_config.MTP_PRECHECK_NEW_TOKENS,**verifier_args)
        if baseline['token_ids']!=speculative['token_ids']:
            raise RuntimeError('Production-checkpoint speculative greedy mismatch')
        checks.append(dict(cached=cached,greedy_exact=True))
    del model,verifier_args
    torch.cuda.empty_cache()
    write_json(e.SANITY_ROOT/'source-correctness.json',checks)
    estimates={}
    for exp in e.collect_experiments():
        metrics=e.SANITY_ROOT/'out'/exp.name/'metrics.jsonl'
        if metrics.exists():metrics.unlink()
        run_logged(e.train_command(exp,probe=True),e.SANITY_ROOT/f'{exp.name}.log',timeout=7200)
        estimates[exp.name]=estimate(exp)
        print(json.dumps(dict(name=exp.name,estimate=estimates[exp.name])),flush=True)
    warmstarts={}
    if mtp.family_config.MTP_UNFREEZE:
        from workloads.speculative.mtp_common import warmstart_metadata
        warmstarts={exp.name:warmstart_metadata(exp.horizons) for exp in e.collect_experiments()}
    result=dict(design_signature=e.SUITE,data=data,warmstarts=warmstarts,
                source_sha256=file_digest(mtp.family_config.MTP_SOURCE),estimates=estimates)
    write_json(e.phase_work('prepare', e.ARTIFACTS_ROOT)/'probe-results.json',result)
    if max(x['projected_training_hours'] for x in estimates.values())>e.MAX_HOURS:
        raise RuntimeError('Measured MTP estimate exceeds approved cap; inspect probe-results.json')
    write_json(e.prepared_path(),result)


def training_inputs(exp, prepared, parent):
    config = mtp.family_config
    return dict(
        experiment=exp.name, horizons=exp.horizons, num_gpus=exp.num_gpus,
        max_horizons=config.MTP_MAX_HORIZONS, step_tokens=config.MTP_STEP_TOKENS,
        source_sha256=prepared["source_sha256"], data_sha256=prepared["data"]["manifest_sha256"],
        warmstart=parent, context=config.MTP_CONTEXT, global_batch=config.MTP_GLOBAL_BATCH,
        micro_batch=config.MTP_MICRO_BATCH, steps=config.MTP_TRAIN_STEPS, seed=config.MTP_SEED,
        train_backbone=config.MTP_UNFREEZE, backend=config.MTP_BACKEND, lr=config.MTP_LR,
        embed_lr=config.MTP_EMBED_LR, warmup=config.MTP_WARMUP_STEPS,
        weight_decay=config.MTP_WEIGHT_DECAY, betas=list(config.MTP_BETAS), grad_clip=config.MTP_GRAD_CLIP,
    )


def check_resume_inputs(previous, current):
    if previous != current:
        raise ValueError("MTP checkpoint training inputs changed")


def train(exp):
    prepared=json.loads((e.prepared_path()).read_text())
    signature=e.SUITE
    if prepared['source_sha256']!=file_digest(mtp.family_config.MTP_SOURCE):raise RuntimeError('Source checkpoint changed')
    if prepared['data']['manifest_sha256']!=file_digest(mtp.family_config.MTP_DATA/'manifest.json'):
        raise RuntimeError('Source data changed')
    check_data()
    parent=None
    if mtp.family_config.MTP_UNFREEZE:
        from workloads.speculative.mtp_common import warmstart_metadata
        parent=warmstart_metadata(exp.horizons)
        if prepared['warmstarts'].get(exp.name)!=parent:raise RuntimeError('Warmstart checkpoint changed')
    forecast=prepared['estimates'][exp.name]
    if forecast['token_budget']!=e.TRAIN_TOKENS or forecast['projected_training_hours']>e.MAX_HOURS:
        raise RuntimeError('Unapproved runtime/budget')
    dest=e.output_dir(exp)
    identity=dest/'run-config.json'
    inputs = training_inputs(exp, prepared, parent)
    if identity.exists():
        check_resume_inputs(json.loads(identity.read_text()).get("training_inputs"), inputs)
    elif any(dest.glob("*.pt")):
        raise RuntimeError("Missing partial-run identity")
    write_json(identity, dict(training_inputs=inputs, command=e.train_command(exp), estimate=forecast))
    start=time.monotonic()
    run_logged(e.train_command(exp),dest/'train.log',timeout=e.MAX_HOURS*3600)
    final=dest/'adaptation_final.pt'
    if not final.is_file():raise RuntimeError('Missing final adaptation checkpoint')
    write_json(dest/'trained.json',dict(checkpoint=str(e.published(final)),design_signature=signature,
        source_sha256=prepared['source_sha256'],
        data_manifest_sha256=prepared['data']['manifest_sha256'],
        seconds=time.monotonic()-start,estimate=forecast,warmstart=parent))


def evaluate(exp):
    dest=e.phase_work('eval', e.output_dir(exp))
    trained=json.loads((e.output_dir(exp)/'trained.json').read_text())
    from workloads.speculative.mtp_common import validate_evaluation_inputs
    validate_evaluation_inputs(trained)
    run_logged([str(e.PYTHON),'scripts/speculative/mtp_evaluate.py','--filter',exp.name],
        dest/'eval.log',timeout=8*3600)
    if not (dest/'evaluation.json').is_file():raise RuntimeError('Missing evaluation results')
    write_json(dest/'evaluated.json',dict(design_signature=e.SUITE,
        results=str(e.published(dest/'evaluation.json'))))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','train','eval'))
    parser.add_argument('--filter');args=parser.parse_args()
    if e.SUITE not in {'mtp','mtp-unfreeze'}:raise RuntimeError('Select an MTP suite')
    if args.phase=='prepare':return prepare()
    matches=e.collect_experiments(args.filter)
    if len(matches)!=1:raise ValueError('Select exactly one experiment')
    (train if args.phase=='train' else evaluate)(matches[0])
