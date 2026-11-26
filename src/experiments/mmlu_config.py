"""MMLU settings for the selected MLSpace worker."""

from .family_arguments import configure_runs
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
experiment_configs = configure_runs(__name__, "experiment_configs", experiment_configs)
