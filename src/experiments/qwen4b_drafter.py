"""Protocol settings for qwen4b drafter."""

from dataclasses import dataclass, field


@dataclass(kw_only=True)
class FamilyConfig:
    DATASET_DIRECTORIES: dict[str, str] = field(default_factory=lambda: {
        "smoltalk-2048": "b0088cb54b3bd646",
        "smoltalk": "smoltalk",
        "dflash-100k": "dflash-100k",
    })
    PHASE_TIMEOUT: dict[str, str] = field(
        default_factory=lambda: {"probe": "40m", "train": "6h", "eval": "3h"}
    )


family_config = FamilyConfig()
