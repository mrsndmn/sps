"""Executable definitions for real readout gsm8k."""

from dataclasses import dataclass, field
from pathlib import Path

from . import real_readout_run as train
from . import vllm_sps_config as c
from .family_arguments import configure, configure_runs, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    ROOT: Path = train.ROOT / "gsm8k-generation"
    HOURS: dict[str, int] = field(
        default_factory=lambda: {"thinking": 48, "nonthinking": 16}
    )
    INSTRUCTION: str = "Solve the problem. End your response with a separate line containing #### followed by the final numeric answer."


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)


experiment_configs = [
    c.Experiment(f"{arm}-{mode}", arm, mode)
    for arm in ("A", "B")
    for mode in family_config.HOURS
]
experiment_configs = configure_runs(__name__, "experiment_configs", experiment_configs)
