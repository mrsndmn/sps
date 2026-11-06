"""Source checkpoint gates, epoch ordering and isolated continuation artifacts."""
import hashlib
import json
import os
from pathlib import Path
from scripts import extend_config as c
os.environ.setdefault('SPS_EXPERIMENT_SUITE', c.SOURCE_SUITE)
import experiments as e


def signature():
    if e.SUITE != c.SOURCE_SUITE or e.design_signature() != c.SOURCE_SIGNATURE:
        raise RuntimeError('Pinned instruct source changed')
    files=sorted(Path(__file__).parent.glob('extend_*.py'))
    return hashlib.sha256(json.dumps(dict(source=c.SOURCE_SIGNATURE,
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),sort_keys=True).encode()).hexdigest()


def source_plan():
    signature()
    plan=json.loads((e.ARTIFACTS_ROOT/'prepared.json').read_text())
    if plan['design_signature']!=c.SOURCE_SIGNATURE or plan['steps']!=c.START_STEP:
        raise RuntimeError('Source plan mismatch')
    return plan


def training_plan():
    return dict(source_plan(),steps=c.TRAIN_STEPS,continuation_from=c.START_STEP,
                data_order='seed-plus-epoch permutation; original epoch zero')


def root(variant=None):
    return e.COMMON_ROOT/c.ROOT_NAME


def arm_name(exp, workload=None):
    return exp.name if workload is None else f'{exp.name}--{workload}'


def output_dir(variant,exp,workload=None):
    name=arm_name(exp,workload)
    if workload is not None: name+=f'--step{variant}'
    return root()/'runs/out'/name


def source_checkpoint(exp):
    return e.output_dir(exp)/'final.pt'


def checkpoint_path(variant,exp):
    return output_dir(variant,exp)/f'step_{int(variant)}.pt'


def validate_state(state,exp,*,source=False,step=None):
    from scripts import instruct_config as original
    expected=c.SOURCE_SIGNATURE if source else signature()
    expected_plan=source_plan() if source else training_plan()
    if (state['design_signature']!=expected or state['source']!=original.MODEL
        or state['revision']!=original.REVISION or state['plan']!=expected_plan
        or state['stage'] is not False or (step is not None and state['step']!=step)
        or (not source and state['experiment']!=exp.__dict__)):
        raise RuntimeError('Continuation checkpoint provenance mismatch')
    if not source and not c.START_STEP <= state['step'] <= c.TRAIN_STEPS:
        raise RuntimeError('Invalid continuation cursor')


def validate_checkpoint(variant,exp,path=None,*,source=False):
    import torch
    path=path or (source_checkpoint(exp) if source else checkpoint_path(variant,exp))
    state=torch.load(path,map_location='meta',mmap=True,weights_only=False)
    validate_state(state,exp,source=source,step=c.START_STEP if source else int(variant))
    return dict(path=str(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,step=state['step'])


from scripts.extend_config import epoch_order, global_batch_indices
