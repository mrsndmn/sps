"""Executable definitions for teacher cap probe."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    caps: tuple[int, ...] = (1024, 1536, 2048, 3072, 4096, 6144, 8192)
    per_source: int = 12
    retention: float = 0.95


experiment_configs = [Probe()]

SHARDS = 8
EVAL_HOURS = 20
