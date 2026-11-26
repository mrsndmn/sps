"""Executable definitions for real readout gsm8k."""

from . import real_readout_run as train, vllm_sps_config as c

ROOT = train.ROOT / "gsm8k-generation"

HOURS = {"thinking": 48, "nonthinking": 16}

INSTRUCTION = "Solve the problem. End your response with a separate line containing #### followed by the final numeric answer."

experiment_configs = [
    c.Experiment(f"{arm}-{mode}", arm, mode) for arm in ("A", "B") for mode in HOURS
]
