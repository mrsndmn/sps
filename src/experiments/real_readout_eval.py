"""Executable definitions for real readout eval."""

from dataclasses import dataclass
from pathlib import Path

from . import ifeval_config as ic
from . import real_readout_run as train
from . import vllm_sps_config as c


@dataclass(kw_only=True)
class FamilyConfig:
    ROOT: Path = train.family_config.ROOT / "endpoint-eval"


family_config = FamilyConfig()

experiment_configs = [
    c.Experiment(f"{arm}-{mode}", arm, mode) for arm in ("A", "B") for mode in ic.family_config.MODES
]
