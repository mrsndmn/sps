"""Executable definitions for real readout serving."""

from dataclasses import dataclass
from pathlib import Path

from . import real_readout_run as train
from . import vllm_sps_config as c


@dataclass(kw_only=True)
class FamilyConfig:
    ROOT: Path = train.family_config.ROOT / "serving"
    LAYOUTS: tuple[str, ...] = ("sps", "sps_mtp")
    AUDIT_PREFIXES: int = 8


family_config = FamilyConfig()

experiment_configs = [c.Experiment(f"serving-{arm}", arm) for arm in ("A", "B")]
