"""Source checkpoint gates, epoch ordering and isolated continuation artifacts."""
from orchestration.named.publication import work_path
import json
import os
from pathlib import Path
from experiments import extend_config as c
os.environ.setdefault('SPS_EXPERIMENT_SUITE', c.family_config.SOURCE_SUITE)
from orchestration import experiment_suite as e


def signature():
    return c.family_config.ROOT_NAME


def source_plan():
    plan=json.loads((e.prepared_path()).read_text())
    if plan['steps'] != c.family_config.START_STEP:
        raise RuntimeError('Source plan mismatch')
    return plan


def training_plan():
    return dict(source_plan(),steps=c.family_config.TRAIN_STEPS,continuation_from=c.family_config.START_STEP,
                data_order='seed-plus-epoch permutation; original epoch zero')


def root(variant=None):
    return e.COMMON_ROOT/c.family_config.ROOT_NAME


def arm_name(exp, workload=None):
    return exp.name if workload is None else f'{exp.name}--{workload}'


def output_dir(variant,exp,workload=None):
    name=arm_name(exp,workload)
    if workload is not None: name+=f'--step{variant}'
    return work_path(root()/'runs/out'/name)


def source_checkpoint(exp):
    return e.output_dir(exp)/'final.pt'


def checkpoint_path(variant,exp):
    return output_dir(variant,exp)/f'step_{int(variant)}.pt'


def validate_state(state,exp,*,source=False,step=None):
    from experiments import instruct_config as original
    expected_plan=source_plan() if source else training_plan()
    if (state['source'] != original.family_config.MODEL or state['revision'] != original.family_config.REVISION or state['plan'] != expected_plan or state['stage'] is not False or step is not None and state['step'] != step or not source and state['experiment'] != exp.__dict__):
        raise RuntimeError('Continuation checkpoint provenance mismatch')
    if not source and not c.family_config.START_STEP <= state['step'] <= c.family_config.TRAIN_STEPS:
        raise RuntimeError('Invalid continuation cursor')


def validate_checkpoint(variant,exp,path=None,*,source=False):
    import torch
    path=path or (source_checkpoint(exp) if source else checkpoint_path(variant,exp))
    state=torch.load(path,map_location='meta',mmap=True,weights_only=False)
    validate_state(state,exp,source=source,step=c.family_config.START_STEP if source else int(variant))
    return dict(path=str(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,step=state['step'])


from experiments.extend_config import epoch_order, global_batch_indices
