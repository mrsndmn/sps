"""Executable definitions for sampling backbone precision."""

from dataclasses import dataclass

from . import sampling_precision as precision
from .family_arguments import configure_runs


@dataclass(frozen=True)
class Experiment(precision.Experiment):
    dtype: str = "float16"


experiment_configs = [
    Experiment("fp16-four", "sps_four_slots", True),
    Experiment("fp16-mtp", "sps_mtp", True),
]
experiment_configs = configure_runs(__name__, "experiment_configs", experiment_configs)


@dataclass(frozen=True)
class Job:
    name: str = "fp16-backbone"
    num_gpus: int = 1
    hours: int = 2


job_configs = [Job()]
job_configs = configure_runs(__name__, "job_configs", job_configs)
