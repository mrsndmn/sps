"""AR-to-SPS settings for the selected MLSpace worker."""

from .ar_sps import ARExperiment, configuration
from .suite import SUITE

globals().update(vars(configuration(SUITE)))
