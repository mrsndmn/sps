"""Isolated rerun provenance and deterministic, optimizer-step LR schedules."""
import hashlib
import json
import math
import os
from pathlib import Path
from scripts import sched_config as c
os.environ.setdefault('SPS_EXPERIMENT_SUITE', c.SOURCE_SUITE)
import experiments as e


def signature():
    if e.SUITE != c.SOURCE_SUITE or e.design_signature() != c.SOURCE_SIGNATURE:
        raise RuntimeError('Pinned instruct source design changed')
    files = sorted(Path(__file__).parent.glob('sched_*.py'))
    return hashlib.sha256(json.dumps(dict(source=c.SOURCE_SIGNATURE,
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}), sort_keys=True).encode()).hexdigest()


def root(variant):
    return e.COMMON_ROOT/c.ROOT_NAMES[variant]


def arm_name(exp, workload=None):
    return exp.name if workload is None else f'{exp.name}--{workload}'


def output_dir(variant, exp, workload=None):
    return root(variant)/'runs/out'/arm_name(exp, workload)


def source_plan():
    signature()
    plan = json.loads((e.ARTIFACTS_ROOT/'prepared.json').read_text())
    if (plan['design_signature'] != c.SOURCE_SIGNATURE or plan['steps'] != c.TRAIN_STEPS
            or plan['stage_steps'] != c.STAGE_STEPS):
        raise RuntimeError('Source plan mismatch')
    return plan


def lr_multiplier(step, total, warmup_fraction=c.WARMUP_FRACTION):
    """Zero-based optimizer update: positive first LR, peak at warmup end, zero last."""
    if not 0 <= step < total or not 0 < warmup_fraction < 1 or total < 2:
        raise ValueError('Invalid schedule coordinates')
    warmup = min(total-1, max(1, math.ceil(total*warmup_fraction)))
    if step < warmup:
        return (step+1)/warmup
    progress = (step+1-warmup)/(total-warmup)
    return c.MIN_LR_RATIO + (1-c.MIN_LR_RATIO)*.5*(1+math.cos(math.pi*progress))


def learning_rate(recipe, step, plan):
    from scripts import instruct_config as source
    if recipe == 'staged_sps':
        split = plan['stage_steps']
        if step < split:
            return source.EMBED_LR*lr_multiplier(step, split)
        return source.FULL_LR*lr_multiplier(step-split, plan['steps']-split)
    return source.FULL_LR*lr_multiplier(step, plan['steps'])


def checkpoint_path(variant, exp):
    return e.output_dir(exp)/'best.pt' if variant == 'step384' else output_dir(variant,exp)/'final.pt'


def validate_checkpoint(variant, exp, path=None):
    import torch
    from scripts import instruct_config as source
    path = path or checkpoint_path(variant,exp)
    state = torch.load(path,map_location='meta',mmap=True,weights_only=False)
    expected = c.SOURCE_SIGNATURE if variant == 'step384' else signature()
    step = c.EVAL_STEP if variant == 'step384' else c.TRAIN_STEPS
    if (state['design_signature'] != expected or state['step'] != step
            or state['source'] != source.MODEL or state['revision'] != source.REVISION
            or state['plan'] != source_plan() or state['stage'] is not False):
        raise RuntimeError(f'Checkpoint provenance mismatch: {path}')
    if variant == 'cosine' and state.get('experiment') != exp.__dict__:
        raise RuntimeError('Checkpoint experiment mismatch')
    return dict(path=str(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,step=step)
