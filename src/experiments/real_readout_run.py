"""Executable definitions for real readout run."""

from dataclasses import dataclass
from pathlib import Path

from . import vllm_sps_config as c


@dataclass(kw_only=True)
class FamilyConfig:
    ROOT: Path = c.family_config.ROOT.parent / "qwen3-instruct-real-readout-mtp"
    PARENT: Path = c.family_config.PARENT.parents[1] / "qwen3_instruct_continued_ar_ce" / "final.pt"
    LEGACY_HORIZONS: int = 4

    @property
    def OUT(self) -> Path:
        return self.ROOT / "runs/out" / experiment_configs[0].name


family_config = FamilyConfig()


experiment_configs = [
    c.Experiment(
        "qwen3_real_readout_mtp", "train", num_gpus=c.experiment_configs[0].num_gpus
    )
]
