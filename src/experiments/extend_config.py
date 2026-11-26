"""Continue constant-LR instruct arms on the same corpus for four epochs."""

import random
from dataclasses import dataclass, field
from functools import lru_cache

from .family_arguments import configure, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    SOURCE_SUITE: str = "ar-sps-qwen3-1.7b-instruct"
    SOURCE_SIGNATURE: str = (
        "761505551994ceec0a907401b3ee875ffea744b69bc94c7007b23c58181b466b"
    )
    START_STEP: int = 512
    TRAIN_STEPS: int = 2048
    EVAL_STEPS: tuple[int, ...] = (1024, 1536, 2048)
    ROOT_NAME: str = "qwen3-instruct-extended-2048"
    CHECKPOINT_INTERVAL: int = 128
    TRAIN_HOURS: int = 8
    WORKLOAD_HOURS: dict[str, int] = field(
        default_factory=lambda: {"ifeval-nonthinking": 12, "ifeval-thinking": 12}
    )
    STARTUP_SECONDS: int = 300

    @property
    def ROOT_NAMES(self) -> dict[str, str]:
        return {str(step): self.ROOT_NAME for step in self.EVAL_STEPS}

    @property
    def WORKLOADS(self) -> tuple[str, ...]:
        return tuple(self.WORKLOAD_HOURS)

    @property
    def BUDGET(self) -> int:
        return 4 * 4 * self.TRAIN_HOURS + 4 * len(self.EVAL_STEPS) * sum(
            self.WORKLOAD_HOURS.values()
        )

    @property
    def BUDGETS(self) -> dict[str, int]:
        return {key: self.BUDGET for key in self.ROOT_NAMES}


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)


@lru_cache(maxsize=4)
def epoch_order(count, epoch, seed):
    order = list(range(count))
    if epoch:
        random.Random(seed + epoch).shuffle(order)
    return tuple(order)


def global_batch_indices(step, count, batch_size, seed):
    if count % batch_size or count < batch_size or step < 0:
        raise ValueError("Dataset must contain full global batches")
    epoch, offset = divmod(step * batch_size, count)
    return epoch_order(count, epoch, seed)[offset : offset + batch_size]
