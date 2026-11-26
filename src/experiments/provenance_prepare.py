"""Executable definitions for provenance prepare."""

from dataclasses import dataclass, field

from . import instruct_config as c
from .family_arguments import configure, configure_runs, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    SPLITS: tuple[str, ...] = ("train", "val", "test")
    GENERATION: dict = field(
        default_factory=lambda: dict(
            temperature=c.GENERATION["nonthinking"]["temperature"],
            top_p=c.GENERATION["nonthinking"]["top_p"],
            top_k=c.GENERATION["nonthinking"]["top_k"],
            max_new_tokens=c.TEACHER_MAX_NEW_TOKENS,
        )
    )
    MIN_PAIRED_FRACTION: float = 0.8


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)


@dataclass(frozen=True)
class Experiment:
    name: str = "paired-response-preparation"
    num_gpus: int = 4
    hours: int = 12


experiment_configs = [Experiment()]
experiment_configs = configure_runs(__name__, "experiment_configs", experiment_configs)
