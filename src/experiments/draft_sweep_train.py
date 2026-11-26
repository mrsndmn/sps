"""Executable definitions for draft sweep train."""

from . import vllm_sps_config as c

ROOT = c.ROOT.parent / "vllm-sps-draft-sweep"

LEGACY_HORIZONS = 9

HORIZONS = LEGACY_HORIZONS

EXPERIMENT = c.experiment_configs[0]
