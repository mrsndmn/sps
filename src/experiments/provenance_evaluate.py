"""Executable definitions for provenance evaluate."""

from dataclasses import dataclass

from . import ifeval_config as ic
from .provenance_train import experiment_configs as training_configs


@dataclass(frozen=True)
class Experiment:
    name: str
    arm: str
    mode: str
    num_gpus: int = 1
    hours: int = ic.family_config.EVAL_HOURS


experiment_configs = [
    Experiment(f"{e.name}--{mode}", e.name, mode)
    for e in training_configs
    for mode in ic.family_config.MODES
]
