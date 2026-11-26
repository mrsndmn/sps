"""Select the job's experiment suite from its environment."""

import os

SUITE = os.environ.get("SPS_EXPERIMENT_SUITE", "1b")
