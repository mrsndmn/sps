"""Protocol settings for dflash repro eval."""

from dataclasses import dataclass

from .family_arguments import configure, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    LIMIT: int = 2048
    WARMUP_TOKENS: int = 32


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)
