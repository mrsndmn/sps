"""Fresh long CE adaptation of Qwen3-1.7B-Base, isolated from pilot artifacts."""
import hashlib
import json
import os
from pathlib import Path

from experiments.base50 import (
    SOURCE_SUITE, ROOT_NAME, STEPS, STAGE_STEPS, FULL_LR, WARMUP_FRACTION,
    CHECKPOINT_INTERVAL, LOG_INTERVAL, JOB_HOURS, REMINDER, learning_rate, experiment_configs,
)


def configure():
    """Bind shared, unchanged pilot utilities only inside this dedicated process."""
    os.environ.setdefault('SPS_EXPERIMENT_SUITE', SOURCE_SUITE)
    from orchestration import experiment_suite as e
    from experiments import ar_sps_config as c
    if e.SUITE != SOURCE_SUITE:
        raise RuntimeError('Run with SPS_EXPERIMENT_SUITE='+SOURCE_SUITE)
    if getattr(e, '_base50_configured', False):
        return e, c
    c.MAX_STEPS = STEPS
    c.MAX_DATA_ROWS = STEPS*c.GLOBAL_BATCH
    c.FULL_LR = FULL_LR
    c.experiment_configs = experiment_configs
    e.experiment_configs = c.experiment_configs
    e.ARTIFACTS_ROOT = e.COMMON_ROOT/ROOT_NAME
    e.RUN_ROOT = e.ARTIFACTS_ROOT/'runs'
    e.SANITY_ROOT = e.ARTIFACTS_ROOT/'sanity'
    e._base50_configured = True
    return e, c
