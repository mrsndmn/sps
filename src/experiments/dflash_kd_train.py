"""Executable definitions for dflash kd train."""

from dataclasses import replace
from .environment import COMMON_ROOT
from . import dflash_repro_train as hard

ROOT = COMMON_ROOT / "dflash-kd"

TEMPERATURE = 1.0

experiment_configs = [
    replace(hard.EXPERIMENT, name="qwen3-4b-dflash-b16-soft-kd", num_gpus=8),
]

(EXPERIMENT,) = experiment_configs
