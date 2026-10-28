"""MLSpace preparation, bounded four-GPU training and matched baseline evaluation."""
import argparse,json,math,shutil,statistics,time
import experiments as e
from scripts.pilot_run import run_logged,write_json
from scripts.spec_baseline_common import identities,checked_state
from scripts.mtp_common import file_sha256,sequence_starts


def check_inputs():
    manifest=json.loads((e.MTP_DATA/'manifest.json').read_text())
    if (manifest.get('revision')!=e.DATASET_REVISION or manifest.get('repo')!=e.DATASET_REPO
            or manifest.get('dtype')!='uint16' or manifest.get('tokenizer')!='gpt2'):
        raise RuntimeError('Unexpected baseline corpus')
    for split in ('train','val'):
        if (e.MTP_DATA/f'{split}.bin').stat().st_size!=2*manifest['tokens'][split]:raise RuntimeError('Incomplete corpus')
    for rank in range(4):
        starts=sequence_starts(manifest['tokens']['train'],e.BASELINE_CONTEXT,4,rank,
            reserve_horizons=e.BASELINE_RESERVE_HORIZONS)
        if len(starts)<e.BASELINE_STEPS*e.BASELINE_GLOBAL_BATCH//4:raise RuntimeError('Insufficient unique inputs')
    if shutil.disk_usage(e.MTP_DATA).free<e.MTP_CHECKPOINT_FREE_BYTES:raise RuntimeError('Insufficient checkpoint disk space')
    return identities()


def estimate(exp):
    rows=[json.loads(x) for x in (e.SANITY_ROOT/'out'/exp.name/'metrics.jsonl').read_text().splitlines()]
    times=[r['step_seconds'] for r in rows if r['event']=='train'][-8:]
    if len(times)!=8 or any(not math.isfinite(x) or x<=0 for x in times):raise RuntimeError('Incomplete baseline probe')
    return dict(seconds_per_step=statistics.median(times),
        estimated_hours_range=[(statistics.median(times)*e.BASELINE_STEPS*1.1+1800)/3600,
                               (max(times)*e.BASELINE_STEPS*1.5+3600)/3600],
        tokens=e.BASELINE_STEPS*e.BASELINE_STEP_TOKENS,probe_steps=e.BASELINE_PROBE_STEPS)


def prepare():
    import torch
    if torch.cuda.device_count()!=4:raise RuntimeError('Preparation requires four visible GPUs')
    identity=check_inputs();e.SANITY_ROOT.mkdir(parents=True,exist_ok=True)
    run_logged([str(e.PYTHON),'-m','pytest','-q','tests/test_spec_baselines.py','tests/test_mtp_speculative.py',
        'tests/test_mtp_parallel.py'],e.SANITY_ROOT/'correctness.log',timeout=3600)
    forecasts={}
    for exp in e.experiment_configs:
        if not exp.trainable:continue
        path=e.SANITY_ROOT/'out'/exp.name/'metrics.jsonl'
        if path.exists():path.unlink()
        run_logged(e.train_command(exp,probe=True),e.SANITY_ROOT/f'{exp.name}.log',timeout=7200)
        forecasts[exp.name]=estimate(exp)
        print(json.dumps(dict(name=exp.name,estimate=forecasts[exp.name])),flush=True)
    controls={}
    for exp in e.experiment_configs:
        if exp.recipe=='existing_mtp':
            trained,_=checked_state(exp)
            controls[exp.name]=dict(path=trained['checkpoint'],sha256=file_sha256(trained['checkpoint']))
    result=dict(design_signature=e.design_signature(),**identity,estimates=forecasts,controls=controls)
    write_json(e.ARTIFACTS_ROOT/'probe-results.json',result)
    if any(v['estimated_hours_range'][1]>e.BASELINE_MAX_HOURS for v in forecasts.values()):
        raise RuntimeError('Baseline forecast exceeds training cap')
    write_json(e.ARTIFACTS_ROOT/'prepared.json',result)


def approved():
    result=json.loads((e.ARTIFACTS_ROOT/'prepared.json').read_text())
    if result['design_signature']!=e.design_signature() or any(result[k]!=v for k,v in check_inputs().items()):
        raise RuntimeError('Changed baseline preparation')
    return result


def train(exp):
    prepared=approved();forecast=prepared['estimates'][exp.name]
    if forecast['estimated_hours_range'][1]>e.BASELINE_MAX_HOURS:raise RuntimeError('Forecast exceeds cap')
    dest=e.output_dir(exp);start=time.monotonic()
    run_logged(e.train_command(exp),dest/'train.log',timeout=e.BASELINE_MAX_HOURS*3600)
    checkpoint=dest/'draft_final.pt'
    import torch
    state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if state['tokens']!=e.BASELINE_STEPS*e.BASELINE_STEP_TOKENS or state['design_signature']!=e.design_signature():
        raise RuntimeError('Training did not complete the intended budget')
    write_json(dest/'trained.json',dict(checkpoint=str(checkpoint),design_signature=e.design_signature(),
        source_sha256=prepared['source_sha256'],data_manifest_sha256=prepared['data_manifest_sha256'],
        seconds=time.monotonic()-start,estimate=forecast))


def evaluate(exp):
    prepared=approved()
    if exp.recipe=='existing_mtp':
        trained,_=checked_state(exp);control=prepared['controls'][exp.name]
        if trained['checkpoint']!=control['path'] or file_sha256(control['path'])!=control['sha256']:
            raise RuntimeError('Changed comparison control checkpoint')
    from scripts.spec_baseline_evaluate import evaluate as run
    import signal
    def expired(signum,frame):raise TimeoutError('Baseline evaluation time cap')
    signal.signal(signal.SIGALRM,expired);signal.alarm(e.BASELINE_EVAL_TIMEOUT)
    try:run(exp)
    finally:signal.alarm(0)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','train','eval'));p.add_argument('--filter');a=p.parse_args()
    if e.SUITE!='spec-baselines':raise RuntimeError('Select spec-baselines suite')
    if a.phase=='prepare':return prepare()
    matches=e.collect_experiments(a.filter)
    if len(matches)!=1:raise ValueError('Select exactly one baseline')
    if a.phase=='train' and not matches[0].trainable:raise ValueError('No training for this baseline')
    (train if a.phase=='train' else evaluate)(matches[0])


if __name__=='__main__':main()
