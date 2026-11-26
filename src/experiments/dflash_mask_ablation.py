"""Executable definitions for dflash mask ablation."""

from dataclasses import dataclass
from .environment import COMMON_ROOT

TARGET = "Qwen/Qwen3-4B"

TARGET_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"

DRAFT = "z-lab/Qwen3-4B-DFlash-b16"

DRAFT_REVISION = "b74e3a329c4d963783143b1e970d95b002be72bd"

ROOT = COMMON_ROOT / "dflash-mask-ablation"

PROMPTS = 24

MAX_NEW_TOKENS = 128

REPEATS = 2


@dataclass(frozen=True)
class Experiment:
    mask: str
    draft_tokens: int

    @property
    def name(self):
        return f"{self.mask}-{self.draft_tokens}"


experiment_configs = [
    Experiment(mask, width) for width in (4, 8) for mask in ("bidirectional", "causal")
]
