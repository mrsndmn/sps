from experiment_defs.speculative_run import ExperimentConfig
from types import SimpleNamespace


def configuration(suite):
    c = SimpleNamespace()
    # Additional proposal baselines. Training budgets count input text tokens once.
    c.BASELINE_CONTEXT = 1024
    c.BASELINE_GLOBAL_BATCH = 64
    c.BASELINE_MICRO_BATCH = 2
    c.BASELINE_STEP_TOKENS = c.BASELINE_CONTEXT * c.BASELINE_GLOBAL_BATCH
    c.BASELINE_STEPS = 15259 + 763
    c.BASELINE_PROBE_STEPS = 16
    c.BASELINE_RESERVE_HORIZONS = 5
    c.BASELINE_MEDUSA_HEADS = 4
    c.BASELINE_LR = {"medusa": 1e-3, "ar_tiny": 3e-4}
    c.BASELINE_WARMUP = 100
    c.BASELINE_WEIGHT_DECAY = 0.01
    c.BASELINE_BETAS = (0.9, 0.95)
    c.BASELINE_CLIP = 1.0
    c.BASELINE_EVAL_INTERVAL = 500
    c.BASELINE_VAL_TOKENS = 131072
    c.BASELINE_MAX_HOURS = 24
    c.BASELINE_AR_ARGS = dict(
        n_layer=4, n_head=4, hidden_size=256, intermediate_size=768
    )
    c.BASELINE_LOOKUP_MATCH = 8
    c.BASELINE_EVAL_HORIZONS = (2, 3, 4)
    c.BASELINE_EVAL_PROMPTS = 64
    c.BASELINE_PROMPT_TOKENS = 128
    c.BASELINE_NEW_TOKENS = 128
    c.BASELINE_WARMUP_PASSES = 2
    c.BASELINE_EVAL_TIMEOUT = 4 * 3600
    if suite == "spec-baselines":
        c.MAX_HOURS = c.BASELINE_MAX_HOURS
        c.TRAIN_TOKENS = c.BASELINE_STEPS * c.BASELINE_STEP_TOKENS
        c.experiment_configs = [
            ExperimentConfig("medusa_heads_ce", "medusa", horizons=4),
            ExperimentConfig("tiny_ar_ce", "ar_tiny", horizons=4),
            ExperimentConfig("history_lookup", "ngram", horizons=4, trainable=False),
            *[
                ExperimentConfig(
                    f"mtp_n{n}_control",
                    "existing_mtp",
                    horizons=n,
                    reference_name=f"s_sps_mtp_n{n}_unfreeze_ce_50m",
                    trainable=False,
                )
                for n in (2, 3, 4)
            ],
        ]

    return c
