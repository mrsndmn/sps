"""Isolated rerun provenance and deterministic, optimizer-step LR schedules."""
from orchestration.named.publication import work_path
import json
import math
import os
from pathlib import Path
from experiments import sched_config as c
os.environ.setdefault('SPS_EXPERIMENT_SUITE', c.family_config.SOURCE_SUITE)
from orchestration import experiment_suite as e


def signature():
    return "qwen3-instruct-cosine"


def root(variant):
    return e.COMMON_ROOT/c.family_config.ROOT_NAMES[variant]


def arm_name(exp, workload=None):
    return exp.name if workload is None else f'{exp.name}--{workload}'


def output_dir(variant, exp, workload=None):
    return work_path(root(variant)/'runs/out'/arm_name(exp, workload))


def source_plan():
    plan = json.loads((e.prepared_path()).read_text())
    if (plan['steps'] != c.family_config.TRAIN_STEPS or plan['stage_steps'] != c.family_config.STAGE_STEPS):
        raise RuntimeError('Source plan mismatch')
    return plan


def lr_multiplier(step, total, warmup_fraction=c.family_config.WARMUP_FRACTION):
    """Zero-based optimizer update: positive first LR, peak at warmup end, zero last."""
    if not 0 <= step < total or not 0 < warmup_fraction < 1 or total < 2:
        raise ValueError('Invalid schedule coordinates')
    warmup = min(total-1, max(1, math.ceil(total*warmup_fraction)))
    if step < warmup:
        return (step+1)/warmup
    progress = (step+1-warmup)/(total-warmup)
    return c.family_config.MIN_LR_RATIO + (1-c.family_config.MIN_LR_RATIO)*.5*(1+math.cos(math.pi*progress))


def learning_rate(recipe, step, plan):
    from experiments import instruct_config as source
    if recipe == 'staged_sps':
        split = plan['stage_steps']
        if step < split:
            return source.family_config.EMBED_LR*lr_multiplier(step, split)
        return source.family_config.FULL_LR*lr_multiplier(step-split, plan['steps']-split)
    return source.family_config.FULL_LR*lr_multiplier(step, plan['steps'])


def checkpoint_path(variant, exp):
    return e.output_dir(exp)/'best.pt' if variant == 'step384' else output_dir(variant,exp)/'final.pt'


def validate_checkpoint(variant, exp, path=None):
    import torch
    from experiments import instruct_config as source
    path = path or checkpoint_path(variant,exp)
    state = torch.load(path,map_location='meta',mmap=True,weights_only=False)
    step = c.family_config.EVAL_STEP if variant == 'step384' else c.family_config.TRAIN_STEPS
    if (state['step'] != step or state['source'] != source.family_config.MODEL or state['revision'] != source.family_config.REVISION or state['plan'] != source_plan() or state['stage'] is not False):
        raise RuntimeError(f'Checkpoint provenance mismatch: {path}')
    if variant == 'cosine' and state.get('experiment') != exp.__dict__:
        raise RuntimeError('Checkpoint experiment mismatch')
    return dict(path=str(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,step=step)
