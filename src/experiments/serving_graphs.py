"""Protocol settings for serving graphs."""

from dataclasses import dataclass, field

from .family_arguments import configure, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    IMPLEMENTATION: str = "31ba03a874f775372154f0edf3e17d436f2d7462"
    COMPILATION: dict = field(
        default_factory=lambda: {
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 3, 4],
        }
    )


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)
