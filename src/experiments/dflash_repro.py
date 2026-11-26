"""Protocol settings for dflash repro."""

from dataclasses import dataclass, field

from .family_arguments import configure, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    TIMEOUTS: dict[str, str] = field(
        default_factory=lambda: {"probe": "90m", "train": "48h", "eval": "12h"}
    )


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)
