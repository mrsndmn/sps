"""Long Base CE adaptation recipe."""

import math
from dataclasses import dataclass

from .types import ARExperiment


@dataclass(kw_only=True)
class FamilyConfig:
    SOURCE_SUITE: str = "ar-sps-qwen3-1.7b"
    ROOT_NAME: str = "qwen3-base-1.7b-ce-50k"
    STEPS: int = 50_000
    STAGE_STEPS: int = 819
    FULL_LR: float = 1e-4
    WARMUP_FRACTION: float = 0.3
    CHECKPOINT_INTERVAL: int = 1000
    LOG_INTERVAL: int = 100
    JOB_HOURS: int = 48
    REMINDER: str = "После успешного завершения напомнить пользователю: он хотел посчитать крамминг на этом чекпоинте."


family_config = FamilyConfig()


def learning_rate(step):
    if not 0 <= step < family_config.STEPS:
        raise ValueError("Step outside training plan")
    if step < family_config.STAGE_STEPS:
        return 1e-2
    total = family_config.STEPS - family_config.STAGE_STEPS
    index = step - family_config.STAGE_STEPS
    warmup = math.ceil(total * family_config.WARMUP_FRACTION)
    if index < warmup:
        return family_config.FULL_LR * (index + 1) / warmup
    return (
        family_config.FULL_LR
        * 0.5
        * (1 + math.cos(math.pi * (index + 1 - warmup) / (total - warmup)))
    )


experiment_configs = [ARExperiment("qwen3_base_staged_sps_ce_50k", "staged_sps")]
