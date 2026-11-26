"""Executable definitions for dflash repro train."""

from dataclasses import dataclass
from pathlib import Path

from .environment import COMMON_ROOT
from .family_arguments import configure, configure_runs, export_fields
from .qwen4b_drafter_train import SEED as SEED


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    ROOT: Path = COMMON_ROOT / "dflash-repro"
    PROPOSALS: int = 15
    MAX_ANCHORS: int = 512
    HEAD_CHUNK: int = 256
    QUERY_BLOCK: int = 128
    LR: float = 6e-4
    WARMUP: float = 0.04
    GAMMA: float = 7.0
    PROBE_WARMUP: int = 6
    PROBE_STEPS: int = 24
    MILESTONES: tuple[int, ...] = (5000, 10000)

    @property
    def WIDTH(self) -> int:
        return self.PROPOSALS + 1


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)


@dataclass(frozen=True)
class Experiment:
    name: str
    output_root: Path
    eval_dir_pattern: str
    epochs: int = 3
    global_batch: int = 24
    num_gpus: int = 4

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
        output_root=family_config.ROOT / "ec224ca5303b0f5b",
        eval_dir_pattern="gsm8k-step-{step:05d}-780d2d2a6375",
    )
]
experiment_configs = configure_runs(__name__, "experiment_configs", experiment_configs)

(EXPERIMENT,) = experiment_configs
