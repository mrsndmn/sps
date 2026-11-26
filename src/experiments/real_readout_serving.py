"""Executable definitions for real readout serving."""

from dataclasses import dataclass
from pathlib import Path

from . import real_readout_run as train
from . import vllm_sps_config as c
from .family_arguments import configure, configure_runs, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    ROOT: Path = train.ROOT / "serving"
    LAYOUTS: tuple[str, ...] = ("sps", "sps_mtp")
    AUDIT_PREFIXES: int = 8


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)

experiment_configs = [c.Experiment(f"serving-{arm}", arm) for arm in ("A", "B")]
experiment_configs = configure_runs(__name__, "experiment_configs", experiment_configs)
