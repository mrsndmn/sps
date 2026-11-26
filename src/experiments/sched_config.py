"""Step-384 evaluation and otherwise matched warmup/cosine instruct reruns."""

from dataclasses import dataclass, field

from .family_arguments import configure, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    SOURCE_SUITE: str = "ar-sps-qwen3-1.7b-instruct"
    SOURCE_SIGNATURE: str = (
        "761505551994ceec0a907401b3ee875ffea744b69bc94c7007b23c58181b466b"
    )
    ROOT_NAMES: dict[str, str] = field(
        default_factory=lambda: {
            "step384": "qwen3-instruct-step384",
            "cosine": "qwen3-instruct-cosine",
        }
    )
    EVAL_STEP: int = 384
    TRAIN_STEPS: int = 512
    STAGE_STEPS: int = 102
    WARMUP_FRACTION: float = 0.3
    MIN_LR_RATIO: float = 0.0
    SCHEDULE_SCOPE: str = "per_stage"
    TRAIN_HOURS: int = 8
    WORKLOAD_HOURS: dict[str, int] = field(
        default_factory=lambda: {
            "quality": 4,
            "ifeval-nonthinking": 12,
            "ifeval-thinking": 12,
            "mmlu-nonthinking": 72,
            "mmlu-thinking": 72,
        }
    )
    STARTUP_SECONDS: int = 300

    @property
    def WORKLOADS(self) -> tuple[str, ...]:
        return tuple(self.WORKLOAD_HOURS)

    @property
    def BUDGETS(self) -> dict[str, int]:
        return {
            "step384": 4 * sum(self.WORKLOAD_HOURS.values()),
            "cosine": 4 * 4 * self.TRAIN_HOURS + 4 * sum(self.WORKLOAD_HOURS.values()),
        }


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)
