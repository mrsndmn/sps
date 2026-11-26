"""Executable definitions for real readout mmlu."""

from . import real_readout_run as train, vllm_sps_config as c

ROOT = train.ROOT / "full-mmlu-generation"

HOURS = 72

COUNT, SUBJECTS = 14042, 57

experiment_configs = [
    c.Experiment(f"{arm}-{mode}", arm, mode)
    for arm in ("A", "B")
    for mode in c.GENERATION
]
