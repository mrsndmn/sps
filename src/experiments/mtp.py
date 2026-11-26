"""MTP settings for the selected MLSpace worker."""

from .mtp_design import configuration
from .family_arguments import configure, export_fields
from .suite import SUITE

family_config = configure(configuration(SUITE), __name__)
experiment_configs = family_config.experiment_configs
__getattr__ = export_fields(family_config)
