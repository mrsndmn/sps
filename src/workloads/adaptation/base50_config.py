"""Fresh long CE adaptation of Qwen3-1.7B-Base, isolated from pilot artifacts."""
import hashlib
import json
import os
from pathlib import Path

from experiments.base50 import family_config as base50_settings
from experiments.base50 import learning_rate, experiment_configs
SOURCE_SUITE = base50_settings.SOURCE_SUITE
ROOT_NAME = base50_settings.ROOT_NAME
STEPS = base50_settings.STEPS
STAGE_STEPS = base50_settings.STAGE_STEPS
FULL_LR = base50_settings.FULL_LR
WARMUP_FRACTION = base50_settings.WARMUP_FRACTION
CHECKPOINT_INTERVAL = base50_settings.CHECKPOINT_INTERVAL
LOG_INTERVAL = base50_settings.LOG_INTERVAL
JOB_HOURS = base50_settings.JOB_HOURS
REMINDER = base50_settings.REMINDER


def configure():
    """Bind shared, unchanged pilot utilities only inside this dedicated process."""
    os.environ.setdefault('SPS_EXPERIMENT_SUITE', SOURCE_SUITE)
    from orchestration import experiment_suite as e
    from experiments import ar_sps_config as c
    if e.SUITE != SOURCE_SUITE:
        raise RuntimeError('Run with SPS_EXPERIMENT_SUITE='+SOURCE_SUITE)
    if getattr(e, '_base50_configured', False):
        return e, c
    c.family_config.MAX_STEPS = STEPS
    c.family_config.MAX_DATA_ROWS = STEPS*c.family_config.GLOBAL_BATCH
    c.family_config.FULL_LR = FULL_LR
    e.experiment_configs = experiment_configs
    e.ARTIFACTS_ROOT = e.COMMON_ROOT/ROOT_NAME
    e.RUN_ROOT = e.ARTIFACTS_ROOT/'runs'
    e.SANITY_ROOT = e.ARTIFACTS_ROOT/'sanity'
    e._base50_configured = True
    return e, c
