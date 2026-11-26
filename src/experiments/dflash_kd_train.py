"""Executable definitions for dflash kd train."""

from dataclasses import dataclass
from pathlib import Path

from . import dflash_repro_train as hard
from .environment import COMMON_ROOT


@dataclass(frozen=True, kw_only=True)
class SoftDistillation(hard.Experiment):
    temperature: float = 1.0


experiment_configs = [
    SoftDistillation(
        name="qwen3-4b-dflash-b16-soft-kd",
        num_gpus=8,
        output_root=COMMON_ROOT / "dflash-kd" / "45d1905c50658e45",
        eval_dir_pattern="gsm8k-step-{step:05d}-5938c51ca9ec",
    ),
]
