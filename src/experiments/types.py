"""Experiment descriptors shared by the local pilot and speculative benchmarks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    recipe: str
    num_gpus: int = 4
    reference_name: str | None = None
    horizons: int = 1
    trainable: bool = True


@dataclass(frozen=True)
class ARExperiment:
    name: str
    recipe: str
    num_gpus: int = 4
    trainable: bool = True
    distillation_weight: float = 0.0
