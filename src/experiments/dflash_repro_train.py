"""Executable definitions for dflash repro train."""

from dataclasses import dataclass, field
from pathlib import Path

from .environment import COMMON_ROOT
from .dflash_repro_eval import GSM8KProtocol, ModelArguments
from .qwen4b_drafter_data import family_config as corpus
from .qwen4b_drafter_train import family_config as qwen4b_drafter_train_settings


@dataclass(frozen=True, kw_only=True)
class TrainArguments(ModelArguments):
    corpus_repo: str = corpus.REPO
    corpus_revision: str = corpus.REVISION
    corpus_rows: int = corpus.SOURCE_ROWS
    name: str
    output_root: Path
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
    milestones: list[int] = field(default_factory=list)
    seed: int = qwen4b_drafter_train_settings.SEED

    @property
    def width(self):
        return self.proposals + 1

    def output(self, phase, step=None):
        if phase == "probe":
            return self.output_root / "probe"
        if phase == "train":
            return self.output_root / "runs/out" / self.name
        raise ValueError(phase)


@dataclass(frozen=True, kw_only=True)
class Experiment(TrainArguments):
    milestones: list[int] = field(default_factory=lambda: [5000, 10000])
    eval_dir_pattern: str
    eval_protocol: GSM8KProtocol = field(default_factory=GSM8KProtocol)

    def output(self, phase, step=None):
        if phase == "eval":
            return (
                self.output_root / "runs/out" / self.eval_dir_pattern.format(step=step)
            )
        return super().output(phase)


BASELINE = Experiment(
    name="qwen3-4b-dflash-b16-scratch",
    output_root=COMMON_ROOT / "dflash-repro" / "ec224ca5303b0f5b",
    eval_dir_pattern="gsm8k-step-{step:05d}-780d2d2a6375",
)

experiment_configs = [BASELINE]


def launches(family):
    from .launch import filtered

    for phase in ("probe", "train", "eval"):
        yield from filtered(
            experiment_configs,
            phase,
            "scripts/distillation/dflash_repro.py",
            "--phase",
            phase,
            options=("step",) if phase == "eval" else (),
        )
