"""AR-to-SPS settings for the selected MLSpace worker."""

from .ar_sps import ARExperiment, configuration
from .family_arguments import configure, configure_runs, export_fields
from .suite import SUITE

family_config = configure(configuration(SUITE), __name__)
experiment_configs = family_config.experiment_configs
experiment_configs = configure_runs(__name__, "experiment_configs", experiment_configs)
__getattr__ = export_fields(family_config)
