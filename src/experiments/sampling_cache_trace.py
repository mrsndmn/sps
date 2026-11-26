"""Executable definitions for sampling cache trace."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    name: str
    arm: str
    cold: bool = False


experiment_configs = [
    Experiment("sequential", "sps"),
    Experiment("speculative", "sps_mtp"),
    Experiment("cold", "sps", True),
]


@dataclass(frozen=True)
class Job:
    name: str = "target-logits"
    num_gpus: int = 1
    hours: int = 2


job_configs = [Job()]
