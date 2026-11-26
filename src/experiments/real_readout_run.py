"""Executable definitions for real readout run."""

from dataclasses import dataclass
from pathlib import Path

from . import vllm_sps_config as c
from .family_arguments import configure, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    ROOT: Path = c.ROOT.parent / "qwen3-instruct-real-readout-mtp"
    PARENT: Path = c.PARENT.parents[1] / "qwen3_instruct_continued_ar_ce" / "final.pt"
    LEGACY_HORIZONS: int = 4

    @property
    def OUT(self) -> Path:
        return self.ROOT / "runs/out" / ARM.name


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)


experiment_configs = [
    c.Experiment(
        "qwen3_real_readout_mtp", "train", num_gpus=c.experiment_configs[0].num_gpus
    )
]

ARM = experiment_configs[0]
