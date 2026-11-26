"""Executable definitions for qwen4b drafter train."""

from dataclasses import dataclass
from .environment import COMMON_ROOT
from .vllm_sps_config import SOURCE_ROOT

ROOT = COMMON_ROOT / "qwen4b-drafter"

DATA = SOURCE_ROOT / "runs" / "data"

LENGTHS = (4, 8)

TARGET_LAYERS = (1, 9, 17, 25, 33)

CONTEXT = 3072

DEFAULT_DATASET = "smoltalk"

PREPARED_DATASET = "dflash-100k"
DATASETS = (DEFAULT_DATASET, PREPARED_DATASET)

ROWS = {"train": 3072, "val": 64}

ANCHORS = 16

EPOCHS = 3

ACCUMULATION = 8

LR = 6e-4

WARMUP = 0.04

GAMMA = 4.0

SEED = 1337

PROBE_STEPS = 3


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
