"""Executable definitions for real readout gsm8k."""

from dataclasses import dataclass, field
from pathlib import Path

from . import real_readout_run as train
from . import vllm_sps_config as c


@dataclass(kw_only=True)
class FamilyConfig:
    ROOT: Path = train.family_config.ROOT / "gsm8k-generation"
    HOURS: dict[str, int] = field(
        default_factory=lambda: {"thinking": 48, "nonthinking": 16}
    )
    INSTRUCTION: str = "Solve the problem. End your response with a separate line containing #### followed by the final numeric answer."


family_config = FamilyConfig()


experiment_configs = [
    c.Experiment(f"{arm}-{mode}", arm, mode)
    for arm in ("A", "B")
    for mode in family_config.HOURS
]
