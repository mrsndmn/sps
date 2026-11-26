"""Fixed-checkpoint self-speculation benchmark definitions."""

from .types import ExperimentConfig

UNIFIED_PROMPTS = 64
UNIFIED_PROMPT_TOKENS = 128
UNIFIED_NEW_TOKENS = 128
UNIFIED_WARMUP_PASSES = 2
UNIFIED_TIMEOUT = 4 * 3600
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
