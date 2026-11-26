"""Executable definitions for provenance train."""

from dataclasses import dataclass

from . import instruct_config as c
from .family_arguments import configure_runs


@dataclass(frozen=True)
class Experiment:
    name: str
    recipe: str
    condition: str
    num_gpus: int = 4
    hours: int = c.TRAIN_HOURS
    trainable: bool = True
    distillation_weight: float = 0.0


experiment_configs = [
    Experiment(f"{recipe}-{condition}", recipe, condition)
    for recipe in ("continued_ar", "staged_sps")
    for condition in ("corpus", "teacher")
]
experiment_configs = configure_runs(__name__, "experiment_configs", experiment_configs)
