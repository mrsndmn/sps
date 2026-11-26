"""DFlash hard-target reproduction runs."""

from dataclasses import dataclass

from .dflash_repro import DFlashConfig
from .environment import COMMON_ROOT

HARD_BASELINE_NAME = "qwen3-4b-dflash-b16-scratch"


@dataclass(frozen=True, kw_only=True)
class HardDFlash(DFlashConfig):
    script = "scripts/distillation/dflash_repro.py"

    @classmethod
    def experiment_configs(cls) -> list["HardDFlash"]:
        return [
            cls(
                name=HARD_BASELINE_NAME,
                output_dir=COMMON_ROOT / "dflash-repro" / "ec224ca5303b0f5b",
                milestones=[5000, 10000],
                archived_eval_pattern="gsm8k-step-{step:05d}-780d2d2a6375",
            ),
        ]
