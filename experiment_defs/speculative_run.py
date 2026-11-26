from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    recipe: str
    num_gpus: int = 4
    reference_name: str | None = None
    horizons: int = 1
    trainable: bool = True
