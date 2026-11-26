"""Allocation for the isolated vLLM runtime setup."""

from dataclasses import dataclass

from .family_arguments import configure_runs


@dataclass(frozen=True)
class RuntimeSetup:
    name: str
    num_gpus: int
    hours: int


experiment_configs = [RuntimeSetup("vllm-sps-main-py312-cu130", num_gpus=1, hours=2)]
experiment_configs = configure_runs(__name__, "experiment_configs", experiment_configs)
