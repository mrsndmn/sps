"""Protocol settings for qwen4b drafter eval."""

from dataclasses import dataclass

from .family_arguments import configure, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    PROMPTS: int = 64
    REPEATS: int = 2
    LIMIT: int = 128
    WARMUP_TOKENS: int = 24


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)
