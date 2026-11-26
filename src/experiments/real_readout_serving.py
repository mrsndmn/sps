"""Executable definitions for real readout serving."""

from . import real_readout_run as train, vllm_sps_config as c

ROOT = train.ROOT / "serving"

experiment_configs = [c.Experiment(f"serving-{arm}", arm) for arm in ("A", "B")]

LAYOUTS = ("sps", "sps_mtp")

AUDIT_PREFIXES = 8
