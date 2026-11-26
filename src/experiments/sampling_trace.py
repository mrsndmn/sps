"""Executable definitions for sampling trace."""

from dataclasses import dataclass

ARMS = ("sps", "sps_four_slots", "sps_mtp")

NATIVE_ARMS = (
    "native-fp32-one",
    "native-fp32-four",
    "native-bf16-one",
    "native-bf16-four",
)


@dataclass(frozen=True)
class Experiment:
    name: str = "divergence-distributions"
    num_gpus: int = 1
    hours: int = 2


experiment_configs = [Experiment()]
