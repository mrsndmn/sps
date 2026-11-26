"""Long Base CE adaptation recipe."""

import math

from .types import ARExperiment

SOURCE_SUITE = "ar-sps-qwen3-1.7b"
ROOT_NAME = "qwen3-base-1.7b-ce-50k"
STEPS = 50_000
STAGE_STEPS = 819
FULL_LR = 1e-4
WARMUP_FRACTION = 0.3
CHECKPOINT_INTERVAL = 1000
LOG_INTERVAL = 100
JOB_HOURS = 48
REMINDER = "После успешного завершения напомнить пользователю: он хотел посчитать крамминг на этом чекпоинте."


def learning_rate(step):
    if not 0 <= step < STEPS:
        raise ValueError("Step outside training plan")
    if step < STAGE_STEPS:
        return 1e-2
    total = STEPS - STAGE_STEPS
    index = step - STAGE_STEPS
    warmup = math.ceil(total * WARMUP_FRACTION)
    if index < warmup:
        return FULL_LR * (index + 1) / warmup
    return (
        FULL_LR
        * 0.5
        * (1 + math.cos(math.pi * (index + 1 - warmup) / (total - warmup)))
    )


experiment_configs = [ARExperiment("qwen3_base_staged_sps_ce_50k", "staged_sps")]
