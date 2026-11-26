"""Executable definitions for qwen4b drafter train."""

from dataclasses import dataclass, field
from pathlib import Path

from .environment import COMMON_ROOT
from .family_arguments import configure, configure_runs, export_fields
from .vllm_sps_config import SOURCE_ROOT


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    ROOT: Path = COMMON_ROOT / "qwen4b-drafter"
    DATA: Path = SOURCE_ROOT / "runs" / "data"
    LENGTHS: tuple[int, ...] = (4, 8)
    TARGET_LAYERS: tuple[int, ...] = (1, 9, 17, 25, 33)
    CONTEXT: int = 3072
    DEFAULT_DATASET: str = "smoltalk"
    PREPARED_DATASET: str = "dflash-100k"
    ROWS: dict[str, int] = field(default_factory=lambda: {"train": 3072, "val": 64})
    ANCHORS: int = 16
    EPOCHS: int = 3
    ACCUMULATION: int = 8
    LR: float = 6e-4
    WARMUP: float = 0.04
    GAMMA: float = 4.0
    SEED: int = 1337
    PROBE_STEPS: int = 3

    @property
    def DATASETS(self) -> tuple[str, ...]:
        return (self.DEFAULT_DATASET, self.PREPARED_DATASET)


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)


@dataclass(frozen=True)
class Experiment:
    mask: str
    embeddings: str

    @property
    def name(self):
        return f"{self.mask}-{self.embeddings}"


experiment_configs = [
    Experiment(mask, embeddings)
    for mask in ("bidirectional", "causal")
    for embeddings in ("shared", "per-position")
]
experiment_configs = configure_runs(__name__, "experiment_configs", experiment_configs)
