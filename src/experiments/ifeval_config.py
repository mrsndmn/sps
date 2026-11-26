"""Parallel IFEval design; reuse the existing instruct experiment registry."""

from dataclasses import dataclass

from .family_arguments import configure, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    SOURCE_SUITE: str = "ar-sps-qwen3-1.7b-instruct"
    SOURCE_SIGNATURE: str = (
        "761505551994ceec0a907401b3ee875ffea744b69bc94c7007b23c58181b466b"
    )
    ARTIFACT_NAME: str = "qwen3-instruct-ifeval"
    MODES: tuple[str, ...] = ("nonthinking", "thinking")
    EVAL_HOURS: int = 12
    STARTUP_SECONDS: int = 300
    GPU_HOURS: int = 125
    LEGACY_SIGNATURE: str = (
        "5336b6a2fc6e9379e4368cd3314a62555546a4156ea2484c19977772335f5896"
    )


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)
