"""Executable definitions for real readout mmlu."""

from dataclasses import dataclass
from pathlib import Path

from . import real_readout_run as train
from . import vllm_sps_config as c


@dataclass(kw_only=True)
class FamilyConfig:
    ROOT: Path = train.family_config.ROOT / "full-mmlu-generation"
    HOURS: int = 72
    COUNT: int = 14042
    SUBJECTS: int = 57


family_config = FamilyConfig()


experiment_configs = [
    c.Experiment(f"{arm}-{mode}", arm, mode)
    for arm in ("A", "B")
    for mode in c.family_config.GENERATION
]
