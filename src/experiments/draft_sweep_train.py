"""Executable definitions for draft sweep train."""

from dataclasses import dataclass
from pathlib import Path

from . import vllm_sps_config as c
from .family_arguments import configure, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    ROOT: Path = c.ROOT.parent / "vllm-sps-draft-sweep"
    LEGACY_HORIZONS: int = 9

    @property
    def HORIZONS(self) -> int:
        return self.LEGACY_HORIZONS


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)


EXPERIMENT = c.experiment_configs[0]
