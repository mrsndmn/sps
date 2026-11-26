"""Executable definitions for dflash kd train."""

from dataclasses import dataclass
from pathlib import Path

from . import dflash_repro_train as hard
from .environment import COMMON_ROOT
from .family_arguments import configure, configure_runs, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    ROOT: Path = COMMON_ROOT / "dflash-kd"
    TEMPERATURE: float = 1.0


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)


experiment_configs = [
    hard.Experiment(
        name="qwen3-4b-dflash-b16-soft-kd",
        num_gpus=8,
        output_root=family_config.ROOT / "45d1905c50658e45",
        eval_dir_pattern="gsm8k-step-{step:05d}-5938c51ca9ec",
    ),
]
experiment_configs = configure_runs(__name__, "experiment_configs", experiment_configs)

(EXPERIMENT,) = experiment_configs
