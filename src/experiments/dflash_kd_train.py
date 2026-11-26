"""DFlash soft-target distillation runs."""

from dataclasses import dataclass

from .dflash_repro import DFlashConfig
from .environment import COMMON_ROOT


@dataclass(frozen=True, kw_only=True)
class SoftKD(DFlashConfig):
    script = "scripts/distillation/dflash_kd.py"
    train_gpus: int = 8
    temperature: float = 1.0

    @classmethod
    def experiment_configs(cls) -> list["SoftKD"]:
        root = COMMON_ROOT / "dflash-kd"
        return [
            cls(
                name="qwen3-4b-dflash-b16-soft-kd",
                output_dir=root / "45d1905c50658e45",
                milestones=[5000, 10000],
                archived_eval_pattern="gsm8k-step-{step:05d}-5938c51ca9ec",
            ),
            cls(
                name="higher-temperature",
                output_dir=root / "higher-temperature",
                temperature=2.0,
                milestones=[5000, 10000],
            ),
        ]
