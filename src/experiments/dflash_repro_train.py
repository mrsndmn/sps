"""Executable definitions for dflash repro train."""

from dataclasses import dataclass
from pathlib import Path

from .environment import COMMON_ROOT
from .qwen4b_drafter_train import family_config as qwen4b_drafter_train_settings


@dataclass(frozen=True, kw_only=True)
class Experiment:
    name: str
    output_root: Path
    eval_dir_pattern: str
    epochs: int = 3
    global_batch: int = 24
    num_gpus: int = 4
    proposals: int = 15
    max_anchors: int = 512
    head_chunk: int = 256
    query_block: int = 128
    lr: float = 6e-4
    warmup: float = 0.04
    gamma: float = 7.0
    probe_warmup: int = 6
    probe_steps: int = 24
    milestones: tuple[int, ...] = (5000, 10000)
    seed: int = qwen4b_drafter_train_settings.SEED

    @property
    def width(self):
        return self.proposals + 1

    def output(self, phase, step=None):
        if phase == "probe":
            return self.output_root / "probe"
        name = (
            self.name if phase == "train" else self.eval_dir_pattern.format(step=step)
        )
        return self.output_root / "runs/out" / name


experiment_configs = [
    Experiment(
        name="qwen3-4b-dflash-b16-scratch",
        output_root=COMMON_ROOT / "dflash-repro" / "ec224ca5303b0f5b",
        eval_dir_pattern="gsm8k-step-{step:05d}-780d2d2a6375",
    )
]
