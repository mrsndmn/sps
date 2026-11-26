"""MTP settings for the selected MLSpace worker."""

from .mtp_design import configuration
from .suite import SUITE

family_config = configuration(SUITE)
experiment_configs = family_config.experiment_configs
