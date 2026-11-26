"""Isolated, bounded Qwen3 instruct adaptation experiment design."""

from dataclasses import dataclass, field

from .family_arguments import configure, configure_runs, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    SUITE: str = "ar-sps-qwen3-1.7b-instruct"
    MODEL: str = "Qwen/Qwen3-1.7B"
    REVISION: str = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    SEED: int = 1337
    CONTEXT: int = 2048
    GLOBAL_BATCH: int = 16
    MICRO_BATCH: int = 1
    MAX_STEPS: int = 512
    MIN_STEPS: int = 32
    STAGE_FRACTION: float = 0.2
    PROBE_STEPS: int = 8
    WARMUP_PROBE: int = 1
    EMBED_LR: float = 1e-2
    FULL_LR: float = 1e-5
    BETAS: tuple[float, ...] = (0.9, 0.95)
    WEIGHT_DECAY: float = 0.01
    CLIP: float = 1.0
    CHECKPOINT_PARTS: int = 4
    DISTILLATION_TEMPERATURE: float = 1.0
    DATASET_REPO: str = "HuggingFaceTB/smoltalk"
    DATASET_REVISION: str = "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"
    DATA_FILES: tuple[str, ...] = tuple(
        f"data/all/train-{i:05d}-of-00009.parquet" for i in range(9)
    )
    DATA_SPLIT_SALT: str = "qwen3-instruct-dialog-v1"
    TRAIN_FRACTION: float = 0.96
    VAL_BOUNDARY: float = 0.98
    THINKING_FRACTION: float = 0.5
    DATA_CANDIDATE_MULTIPLIER: int = 2
    MIN_DATA_FRACTION: float = 0.5
    TEACHER_MAX_NEW_TOKENS: int = 1024
    TEACHER_TEMPERATURE: float = 0.6
    TEACHER_TOP_P: float = 0.95
    TEACHER_TOP_K: int = 20
    MMLU_DATASET: str = "cais/mmlu"
    MMLU_REVISION: str = "c30699e8356da336a370243923dbaf21066bb9fe"
    IFEVAL_DATASET: str = "google/IFEval"
    IFEVAL_REVISION: str = "966cd89545d6b6acfd7638bc708b98261ca58e84"
    GENERATION: dict[str, dict] = field(
        default_factory=lambda: {
            "thinking": dict(
                temperature=0.6, top_p=0.95, top_k=20, max_new_tokens=2048
            ),
            "nonthinking": dict(
                temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=512
            ),
        }
    )
    GENERATION_AUDIT_COUNT: int = 8
    AUDIT_LENGTH: int = 32
    GENERATION_PARITY_ATOL: float = 2e-4
    GENERATION_PARITY_RTOL: float = 2e-4
    LIKELIHOOD_CONTEXT: int = 512
    LIKELIHOOD_TASKS: tuple[str, ...] = (
        "arc_easy",
        "hellaswag",
        "piqa",
        "sciq",
        "lambada_openai",
    )
    EVAL_BATCH: int = 1
    PREPARE_HOURS: int = 24
    TRAIN_HOURS: int = 8
    EVAL_HOURS: int = 72
    STARTUP_SECONDS: int = 300
    SAFETY: float = 2.0

    @property
    def DATA_ROWS(self) -> dict[str, int]:
        return {"train": self.MAX_STEPS * self.GLOBAL_BATCH, "val": 128, "test": 128}

    @property
    def CONTAMINATION_DATASETS(self) -> tuple[dict, ...]:
        return (
            dict(
                repo=self.MMLU_DATASET,
                config="all",
                revision=self.MMLU_REVISION,
                split="test",
                fields=("question",),
            ),
            dict(
                repo=self.IFEVAL_DATASET,
                revision=self.IFEVAL_REVISION,
                split="train",
                fields=("prompt",),
            ),
            dict(
                repo="allenai/ai2_arc",
                config="ARC-Easy",
                revision="210d026faf9955653af8916fad021475a3f00453",
                split="test",
                fields=("question",),
            ),
            dict(
                repo="Rowan/hellaswag",
                revision="218ec52e09a7e7462a5400043bb9a69a41d06b76",
                split="validation",
                fields=("ctx",),
            ),
            dict(
                repo="baber/piqa",
                revision="142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
                split="validation",
                fields=("goal",),
            ),
            dict(
                repo="allenai/sciq",
                revision="2c94ad3e1aafab77146f384e23536f97a4849815",
                split="test",
                fields=("question",),
            ),
            dict(
                repo="EleutherAI/lambada_openai",
                config="default",
                revision="900124bf3b8235c6daf21033af9948b3f07346c4",
                split="test",
                fields=("text",),
                trust_remote_code=True,
            ),
        )

    @property
    def LIKELIHOOD_DATASETS(self) -> dict[str, dict]:
        return dict(zip(self.LIKELIHOOD_TASKS, self.CONTAMINATION_DATASETS[2:]))

    @property
    def EVAL_MAX_HOURS(self) -> int:
        return self.EVAL_HOURS - 1

    @property
    def GPU_HOURS(self) -> int:
        return (
            self.PREPARE_HOURS * 4
            + self.TRAIN_HOURS * 4 * 4
            + self.EVAL_HOURS * len(experiment_configs)
        )


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)
# Queue waiting is not charged; these are bounded worker execution reservations.


@dataclass(frozen=True)
class InstructExperiment:
    name: str
    recipe: str
    num_gpus: int = 4
    trainable: bool = True
    distillation_weight: float = 0.0


experiment_configs = [
    InstructExperiment("qwen3_instruct_original_ar", "original_ar", trainable=False),
    InstructExperiment("qwen3_instruct_staged_sps_ce", "staged_sps"),
    InstructExperiment(
        "qwen3_instruct_staged_sps_kd", "staged_sps", distillation_weight=0.5
    ),
    InstructExperiment("qwen3_instruct_continued_ar_ce", "continued_ar"),
    InstructExperiment(
        "qwen3_instruct_continued_ar_kd", "continued_ar", distillation_weight=0.5
    ),
]
experiment_configs = configure_runs(__name__, "experiment_configs", experiment_configs)
