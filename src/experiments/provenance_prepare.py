"""Executable definitions for provenance prepare."""

from dataclasses import dataclass, field

from . import instruct_config as c


@dataclass(kw_only=True)
class FamilyConfig:
    SPLITS: tuple[str, ...] = ("train", "val", "test")
    GENERATION: dict = field(
        default_factory=lambda: dict(
            temperature=c.family_config.GENERATION["nonthinking"]["temperature"],
            top_p=c.family_config.GENERATION["nonthinking"]["top_p"],
            top_k=c.family_config.GENERATION["nonthinking"]["top_k"],
            max_new_tokens=c.family_config.TEACHER_MAX_NEW_TOKENS,
        )
    )
    MIN_PAIRED_FRACTION: float = 0.8


family_config = FamilyConfig()


@dataclass(frozen=True)
class Experiment:
    name: str = "paired-response-preparation"
    num_gpus: int = 4
    hours: int = 12


experiment_configs = [Experiment()]
