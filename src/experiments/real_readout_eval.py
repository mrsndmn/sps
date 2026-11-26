"""Executable definitions for real readout eval."""

from dataclasses import dataclass
from pathlib import Path

from . import ifeval_config as ic
from . import real_readout_run as train
from . import vllm_sps_config as c
from .family_arguments import configure, configure_runs, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    ROOT: Path = train.ROOT / "endpoint-eval"


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)

experiment_configs = [
    c.Experiment(f"{arm}-{mode}", arm, mode) for arm in ("A", "B") for mode in ic.MODES
]
experiment_configs = configure_runs(__name__, "experiment_configs", experiment_configs)
