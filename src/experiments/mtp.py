"""SPS multi-token adaptation and verifier definitions."""

from .environment import COMMON_ROOT
from .suite import SUITE
from .types import ExperimentConfig

MTP_SOURCE = COMMON_ROOT / "10b/runs/out/s_sps_w0_10b/ckpt_tokens_10003415040_final.pt"
MTP_DATA = COMMON_ROOT / "10b/runs/data/fineweb-edu"
MTP_CONTEXT = 1024
MTP_MICRO_BATCH = 2
MTP_GLOBAL_BATCH = 64
MTP_STEP_TOKENS = MTP_CONTEXT * MTP_GLOBAL_BATCH
MTP_TRAIN_STEPS = (1_000_000_000 + MTP_STEP_TOKENS - 1) // MTP_STEP_TOKENS
MTP_PROBE_STEPS = 16
MTP_LR = 0.01
MTP_WARMUP_STEPS = 100
MTP_WEIGHT_DECAY = 0.0
MTP_BETAS = (0.9, 0.95)
MTP_GRAD_CLIP = 1.0
MTP_EVAL_INTERVAL = 500
MTP_VAL_TOKENS = 131072
MTP_EVAL_PROMPTS = 16
MTP_PROMPT_TOKENS = 128
MTP_NEW_TOKENS = 128
MTP_TEMPERATURE = 1.0
MTP_SEED = 1337
MTP_BACKEND = "auto"
MTP_INFERENCE_BACKEND = "sdpa"
MTP_MAX_HORIZONS = 4
MTP_UNFREEZE_MAX_HOURS = 4
MTP_UNFREEZE = SUITE in {"mtp-unfreeze", "mtp-parallel"}
MTP_PARENT_ROOT = COMMON_ROOT / "mtp/runs/out"
MTP_PARENT_INPUT_TOKENS = MTP_TRAIN_STEPS * MTP_STEP_TOKENS
MTP_PARENT_STEP_TOKENS = MTP_STEP_TOKENS
MTP_PARENT_CONTEXT = MTP_CONTEXT
MTP_EMBED_LR = MTP_LR
MTP_CHECKPOINT_FREE_BYTES = 20_000_000_000
MTP_PRECHECK_PROMPTS = 2
MTP_PRECHECK_NEW_TOKENS = 16
if MTP_UNFREEZE:
    MTP_TRAIN_STEPS = (50_000_000 + MTP_STEP_TOKENS - 1) // MTP_STEP_TOKENS
    MTP_LR = 1e-5
    MTP_EMBED_LR = 1e-4
    MTP_WARMUP_STEPS = 30
    MTP_WEIGHT_DECAY = 0.01
    MTP_EVAL_INTERVAL = (10_000_000 + MTP_STEP_TOKENS - 1) // MTP_STEP_TOKENS
    MTP_EVAL_PROMPTS = 64
if SUITE in {"mtp", "mtp-unfreeze", "mtp-parallel"}:
    experiment_configs = [
        ExperimentConfig(f"s_sps_mtp_n{n}_emb_1b", "s_sps_w0_20b", horizons=n)
        for n in (2, 3, 4)
    ]
    if MTP_UNFREEZE:
        experiment_configs = [
            ExperimentConfig(
                f"s_sps_mtp_n{n}_unfreeze_ce_50m", "s_sps_w0_20b", horizons=n
            )
            for n in (2, 3, 4)
        ]

MTP_PARALLEL_PROB_ATOL = 0.005
MTP_PARALLEL_AUDIT_PROMPTS = 8

MTP_PARALLEL_TIMEOUT_SECONDS = 4 * 3600
