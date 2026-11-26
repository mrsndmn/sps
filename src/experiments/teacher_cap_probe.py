"""Executable definitions for teacher cap probe."""

from dataclasses import dataclass

from .family_arguments import configure, export_fields


@dataclass(frozen=True)
class Probe:
    caps: tuple[int, ...] = (1024, 1536, 2048, 3072, 4096, 6144, 8192)
    per_source: int = 12
    retention: float = 0.95


experiment_configs = [Probe()]


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    OUTPUT_NAME: str = "5bb50eb67bff7ec297e740ea8bffdce6d220f9bf5051cc47d67a5755634f82c5"
    SHARDS: int = 8
    EVAL_HOURS: int = 20


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)
