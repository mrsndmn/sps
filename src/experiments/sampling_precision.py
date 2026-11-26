"""Executable definitions for sampling precision."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    name: str
    arm: str
    fp32_head: bool


experiment_configs = [
    Experiment("bf16-four", "sps_four_slots", False),
    Experiment("fp32head-one", "sps", True),
    Experiment("fp32head-four", "sps_four_slots", True),
    Experiment("fp32head-mtp", "sps_mtp", True),
]


@dataclass(frozen=True)
class Job:
    name: str = "matched-slots-and-head"
    num_gpus: int = 1
    hours: int = 2


job_configs = [Job()]
