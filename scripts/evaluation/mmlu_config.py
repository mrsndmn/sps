"""Settings used by the generative MMLU workers."""
import os
from experiment_defs.mmlu import (DATASET, REVISION, MAX_NEW_TOKENS, MAX_HOURS, SEED,
                                  Evaluation, SOURCES, FULL_RESUME_EXCLUDED, experiments)

experiment_configs = experiments(os.environ.get("SPS_EXPERIMENT_SUITE"))
