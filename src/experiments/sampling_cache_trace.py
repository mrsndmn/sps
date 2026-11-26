"""Executable definitions for sampling cache trace."""

from dataclasses import dataclass

from .family_arguments import configure_runs


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
experiment_configs = configure_runs(__name__, "experiment_configs", experiment_configs)


@dataclass(frozen=True)
class Job:
    name: str = "target-logits"
    num_gpus: int = 1
    hours: int = 2


job_configs = [Job()]
job_configs = configure_runs(__name__, "job_configs", job_configs)
