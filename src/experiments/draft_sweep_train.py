"""Executable definitions for draft sweep train."""

from dataclasses import dataclass
from pathlib import Path

from . import vllm_sps_config as c
from .launch import named_run


@dataclass(kw_only=True)
class FamilyConfig:
    ROOT: Path = c.family_config.ROOT.parent / "vllm-sps-draft-sweep"
    LEGACY_HORIZONS: int = 9

    @property
    def HORIZONS(self) -> int:
        return self.LEGACY_HORIZONS


family_config = FamilyConfig()


EXPERIMENT = named_run(c.experiment_configs, c.SPS_MTP.name)
