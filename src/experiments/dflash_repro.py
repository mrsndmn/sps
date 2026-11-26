"""Protocol settings for dflash repro."""

from dataclasses import dataclass, field


@dataclass(kw_only=True)
class FamilyConfig:
    TIMEOUTS: dict[str, str] = field(
        default_factory=lambda: {"probe": "90m", "train": "48h", "eval": "12h"}
    )


family_config = FamilyConfig()
