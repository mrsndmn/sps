"""MMLU settings for the selected MLSpace worker."""

from .mmlu import family_config as mmlu_settings
from .mmlu import Evaluation, experiments
DATASET = mmlu_settings.DATASET
FULL_RESUME_EXCLUDED = mmlu_settings.FULL_RESUME_EXCLUDED
MAX_HOURS = mmlu_settings.MAX_HOURS
MAX_NEW_TOKENS = mmlu_settings.MAX_NEW_TOKENS
REVISION = mmlu_settings.REVISION
SEED = mmlu_settings.SEED
SOURCES = mmlu_settings.SOURCES
from .suite import SUITE

experiment_configs = experiments(SUITE)
