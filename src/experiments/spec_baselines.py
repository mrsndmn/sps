"""Medusa, tiny AR and history-lookup baseline definitions."""

from .types import ExperimentConfig

BASELINE_CONTEXT = 1024
BASELINE_GLOBAL_BATCH = 64
BASELINE_MICRO_BATCH = 2
BASELINE_STEP_TOKENS = BASELINE_CONTEXT * BASELINE_GLOBAL_BATCH
BASELINE_STEPS = 15259 + 763
BASELINE_PROBE_STEPS = 16
BASELINE_RESERVE_HORIZONS = 5
BASELINE_MEDUSA_HEADS = 4
BASELINE_LR = {"medusa": 1e-3, "ar_tiny": 3e-4}
BASELINE_WARMUP = 100
BASELINE_WEIGHT_DECAY = 0.01
BASELINE_BETAS = (0.9, 0.95)
BASELINE_CLIP = 1.0
BASELINE_EVAL_INTERVAL = 500
BASELINE_VAL_TOKENS = 131072
BASELINE_MAX_HOURS = 24
BASELINE_AR_ARGS = dict(n_layer=4, n_head=4, hidden_size=256, intermediate_size=768)
BASELINE_LOOKUP_MATCH = 8
BASELINE_EVAL_HORIZONS = (2, 3, 4)
BASELINE_EVAL_PROMPTS = 64
BASELINE_PROMPT_TOKENS = 128
BASELINE_NEW_TOKENS = 128
BASELINE_WARMUP_PASSES = 2
BASELINE_EVAL_TIMEOUT = 4 * 3600
experiment_configs = [
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
