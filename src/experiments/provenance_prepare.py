"""Executable definitions for provenance prepare."""

from dataclasses import dataclass
from . import instruct_config as c

SPLITS = ("train", "val", "test")

GENERATION = dict(
    temperature=c.GENERATION["nonthinking"]["temperature"],
    top_p=c.GENERATION["nonthinking"]["top_p"],
    top_k=c.GENERATION["nonthinking"]["top_k"],
    max_new_tokens=c.TEACHER_MAX_NEW_TOKENS,
)

MIN_PAIRED_FRACTION = 0.8


@dataclass(frozen=True)
class Experiment:
    name: str = "paired-response-preparation"
    num_gpus: int = 4
    hours: int = 12


experiment_configs = [Experiment()]
