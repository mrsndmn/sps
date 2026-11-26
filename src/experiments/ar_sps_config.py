"""AR-to-SPS settings for the selected MLSpace worker."""

from .ar_sps import ARExperiment, configuration
from .suite import SUITE

family_config = configuration(SUITE)
experiment_configs = family_config.experiment_configs
