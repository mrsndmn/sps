from experiment_defs.speculative_run import ExperimentConfig
from types import SimpleNamespace


def configuration(suite):
    c = SimpleNamespace()
    # Unified self-speculation consumes full continuations; no additional training.
    c.UNIFIED_PROMPTS = 64
    c.UNIFIED_PROMPT_TOKENS = 128
    c.UNIFIED_NEW_TOKENS = 128
    c.UNIFIED_WARMUP_PASSES = 2
    c.UNIFIED_TIMEOUT = 4 * 3600
    if suite == "mtp-unified":
        c.experiment_configs = [
            ExperimentConfig(
                f"mtp_n{n}_self",
                "existing_mtp",
                horizons=n,
                reference_name=f"s_sps_mtp_n{n}_unfreeze_ce_50m",
                trainable=False,
            )
            for n in (2, 3, 4)
        ]

    return c
