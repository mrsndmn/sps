"""Executable definitions for real readout eval."""

from . import ifeval_config as ic, real_readout_run as train, vllm_sps_config as c

ROOT = train.ROOT / "endpoint-eval"

experiment_configs = [
    c.Experiment(f"{arm}-{mode}", arm, mode) for arm in ("A", "B") for mode in ic.MODES
]
