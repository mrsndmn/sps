"""MMLU settings for the selected MLSpace worker."""

from .mmlu import (
    DATASET,
    FULL_RESUME_EXCLUDED,
    MAX_HOURS,
    MAX_NEW_TOKENS,
    REVISION,
    SEED,
    SOURCES,
    Evaluation,
    experiments,
)
from .suite import SUITE

experiment_configs = experiments(SUITE)
