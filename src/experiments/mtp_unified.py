"""Fixed-checkpoint self-speculation benchmark definitions."""

from dataclasses import dataclass

from .family_arguments import configure, configure_runs, export_fields
from .types import ExperimentConfig


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    UNIFIED_PROMPTS: int = 64
    UNIFIED_PROMPT_TOKENS: int = 128
    UNIFIED_NEW_TOKENS: int = 128
    UNIFIED_WARMUP_PASSES: int = 2
    UNIFIED_TIMEOUT: int = 4 * 3600


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)
experiment_configs = [
    ExperimentConfig(
        f"mtp_n{n}_self",
        "existing_mtp",
        horizons=n,
        reference_name=f"s_sps_mtp_n{n}_unfreeze_ce_50m",
        trainable=False,
    )
    for n in (2, 3, 4)
]
experiment_configs = configure_runs(__name__, "experiment_configs", experiment_configs)
