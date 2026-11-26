"""DFlash experiment settings, phase directories and worker argument schemas."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from .dflash_mask_ablation import family_config as models
from .qwen4b_drafter_data import family_config as dataset
from .qwen4b_drafter_train import family_config as corpus
from .launch import filtered


@dataclass(frozen=True, kw_only=True)
class ModelArguments:
    target: str = models.TARGET
    target_revision: str = models.TARGET_REVISION
    draft: str = models.DRAFT
    draft_revision: str = models.DRAFT_REVISION
    selection_file: Path = corpus.SELECTION_FILE


@dataclass(frozen=True, kw_only=True)
class GSM8KProtocol:
    gsm8k: str = "openai/gsm8k"
    gsm8k_revision: str = "740312add88f781978c0658806c59bc2815b9866"
    gsm8k_file: str = "main/test-00000-of-00001.parquet"
    limit: int = 2048
    warmup_tokens: int = 32


@dataclass(frozen=True, kw_only=True)
class DFlashEvalArguments(GSM8KProtocol, ModelArguments):
    name: str
    checkpoint: Path
    result: Path
    proposals: int
    seed: int
    training_settings_sha256: str

    @property
    def width(self):
        return self.proposals + 1


@dataclass(frozen=True, kw_only=True)
class TrainingSettings(ModelArguments):
    corpus_repo: str = dataset.REPO
    corpus_revision: str = dataset.REVISION
    corpus_rows: int = dataset.SOURCE_ROWS
    name: str
    epochs: int = 3
    global_batch: int = 24
    train_gpus: int = 4
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
    seed: int = corpus.SEED

    @property
    def width(self):
        return self.proposals + 1


@dataclass(frozen=True, kw_only=True)
class TrainArguments(TrainingSettings):
    result_dir: Path


@dataclass(frozen=True, kw_only=True)
class SoftTrainArguments(TrainArguments):
    temperature: float


@dataclass(frozen=True, kw_only=True)
class DFlashConfig(TrainingSettings):
    output_dir: Path
    eval_protocol: GSM8KProtocol = field(default_factory=GSM8KProtocol)
    timeouts: dict[str, str] = field(default_factory=lambda: {"probe": "90m", "train": "48h", "eval": "12h"})
    # The two published baselines use the original runs/out layout.
    archived_eval_pattern: str | None = None

    script: ClassVar[str]
    phases: ClassVar[tuple[str, ...]] = ("probe", "train", "eval")

    def output(self, phase, step=None):
        if phase == "probe":
            return self.output_dir / "probe"
        if phase == "train":
            if self.archived_eval_pattern:
                return self.output_dir / "runs/out" / self.name
            return self.output_dir / "train"
        if phase == "eval" and step is not None:
            if self.archived_eval_pattern:
                return self.output_dir / "runs/out" / self.archived_eval_pattern.format(step=step)
            return self.output_dir / "eval" / f"step-{step:05d}"
        raise ValueError(f"Invalid DFlash phase or step: {phase}, {step}")

    @classmethod
    def launches(cls, family):
        configs = cls.experiment_configs()
        for phase in cls.phases:
            yield from filtered(configs, phase, cls.script, "--phase", phase,
                                options=("step",) if phase == "eval" else ())
