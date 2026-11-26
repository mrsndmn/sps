"""Parallel IFEval design; reuse the existing instruct experiment registry."""

from dataclasses import dataclass

from .family_arguments import configure, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    SOURCE_SUITE: str = "ar-sps-qwen3-1.7b-instruct"
    ARTIFACT_NAME: str = "qwen3-instruct-ifeval"
    MODES: tuple[str, ...] = ("nonthinking", "thinking")
    EVAL_HOURS: int = 12
    STARTUP_SECONDS: int = 300
    GPU_HOURS: int = 125


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)
