"""Executable definitions for real readout mmlu."""

from dataclasses import dataclass
from pathlib import Path

from . import real_readout_run as train
from . import vllm_sps_config as c
from .family_arguments import configure, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    ROOT: Path = train.ROOT / "full-mmlu-generation"
    HOURS: int = 72
    COUNT: int = 14042
    SUBJECTS: int = 57


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)


experiment_configs = [
    c.Experiment(f"{arm}-{mode}", arm, mode)
    for arm in ("A", "B")
    for mode in c.GENERATION
]
