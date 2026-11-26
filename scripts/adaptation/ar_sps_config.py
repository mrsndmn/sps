"""Suite-specific settings for legacy adaptation workers."""
import os

from experiment_defs.ar_sps import ARExperiment, configuration

globals().update(vars(configuration(os.environ.get("SPS_EXPERIMENT_SUITE"))))
