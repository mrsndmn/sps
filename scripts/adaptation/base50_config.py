"""Fresh long CE adaptation of Qwen3-1.7B-Base, isolated from pilot artifacts."""
import hashlib
import json
import math
import os
from pathlib import Path

SOURCE_SUITE = 'ar-sps-qwen3-1.7b'
ROOT_NAME = 'qwen3-base-1.7b-ce-50k'
STEPS = 50_000
STAGE_STEPS = 819
FULL_LR = 1e-4
WARMUP_FRACTION = .3
CHECKPOINT_INTERVAL = 1000
LOG_INTERVAL = 100
JOB_HOURS = 48
REMINDER = 'После успешного завершения напомнить пользователю: он хотел посчитать крамминг на этом чекпоинте.'


def learning_rate(step):
    if not 0 <= step < STEPS:
        raise ValueError('Step outside training plan')
    if step < STAGE_STEPS:
        return 1e-2
    total = STEPS-STAGE_STEPS
    index = step-STAGE_STEPS
    warmup = math.ceil(total*WARMUP_FRACTION)
    if index < warmup:
        return FULL_LR*(index+1)/warmup
    return FULL_LR*.5*(1+math.cos(math.pi*(index+1-warmup)/(total-warmup)))


def configure():
    """Bind shared, unchanged pilot utilities only inside this dedicated process."""
    os.environ.setdefault('SPS_EXPERIMENT_SUITE', SOURCE_SUITE)
    import experiments as e
    from scripts import ar_sps_config as c
    if e.SUITE != SOURCE_SUITE:
        raise RuntimeError('Run with SPS_EXPERIMENT_SUITE='+SOURCE_SUITE)
    if getattr(e, '_base50_configured', False):
        return e, c
    original_signature = e.design_signature()
    files = sorted(Path(__file__).parent.glob('base50_*.py'))
    signature = hashlib.sha256(json.dumps(dict(source=original_signature,
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),sort_keys=True).encode()).hexdigest()
    c.MAX_STEPS = STEPS
    c.MAX_DATA_ROWS = STEPS*c.GLOBAL_BATCH
    c.FULL_LR = FULL_LR
    c.experiment_configs = [c.ARExperiment('qwen3_base_staged_sps_ce_50k', 'staged_sps')]
    e.experiment_configs = c.experiment_configs
    e.ARTIFACTS_ROOT = e.COMMON_ROOT/ROOT_NAME
    e.RUN_ROOT = e.ARTIFACTS_ROOT/'runs'
    e.SANITY_ROOT = e.ARTIFACTS_ROOT/'sanity'
    e.design_signature = lambda: signature
    e._base50_configured = True
    return e, c
