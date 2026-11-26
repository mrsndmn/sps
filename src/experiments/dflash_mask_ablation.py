"""Executable definitions for dflash mask ablation."""

from dataclasses import dataclass
from pathlib import Path

from .environment import COMMON_ROOT
from .family_arguments import configure, configure_runs, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    OUTPUT_NAME: str = "277ae900c11f7e9a"
    TARGET: str = "Qwen/Qwen3-4B"
    TARGET_REVISION: str = "1cfa9a7208912126459214e8b04321603b3df60c"
    DRAFT: str = "z-lab/Qwen3-4B-DFlash-b16"
    DRAFT_REVISION: str = "b74e3a329c4d963783143b1e970d95b002be72bd"
    ROOT: Path = COMMON_ROOT / "dflash-mask-ablation"
    PROMPTS: int = 24
    MAX_NEW_TOKENS: int = 128
    REPEATS: int = 2


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)


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
experiment_configs = configure_runs(__name__, "experiment_configs", experiment_configs)
