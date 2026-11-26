"""Allocation for the isolated vLLM runtime setup."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeSetup:
    name: str
    num_gpus: int
    hours: int


experiment_configs = [RuntimeSetup("vllm-sps-main-py312-cu130", num_gpus=1, hours=2)]
