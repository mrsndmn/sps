from experiment_defs.paths import COMMON_ROOT
from experiment_defs.speculative_run import ExperimentConfig
from types import SimpleNamespace


def configuration(suite):
    c = SimpleNamespace()
    # Embedding-only adaptation: source settings and benchmark workloads live here.
    c.MTP_SOURCE = (
        COMMON_ROOT / "10b/runs/out/s_sps_w0_10b/ckpt_tokens_10003415040_final.pt"
    )
    c.MTP_DATA = COMMON_ROOT / "10b/runs/data/fineweb-edu"
    c.MTP_CONTEXT = 1024
    c.MTP_MICRO_BATCH = 2
    c.MTP_GLOBAL_BATCH = 64
    c.MTP_STEP_TOKENS = c.MTP_CONTEXT * c.MTP_GLOBAL_BATCH
    c.MTP_TRAIN_STEPS = (1_000_000_000 + c.MTP_STEP_TOKENS - 1) // c.MTP_STEP_TOKENS
    c.MTP_PROBE_STEPS = 16
    c.MTP_LR = 0.01
    c.MTP_WARMUP_STEPS = 100
    c.MTP_WEIGHT_DECAY = 0.0
    c.MTP_BETAS = (0.9, 0.95)
    c.MTP_GRAD_CLIP = 1.0
    c.MTP_EVAL_INTERVAL = 500
    c.MTP_VAL_TOKENS = 131072
    c.MTP_EVAL_PROMPTS = 16
    c.MTP_PROMPT_TOKENS = 128
    c.MTP_NEW_TOKENS = 128
    c.MTP_TEMPERATURE = 1.0
    c.MTP_SEED = 1337
    c.MTP_BACKEND = "auto"
    c.MTP_INFERENCE_BACKEND = "sdpa"
    c.MTP_MAX_HORIZONS = 4
    c.MTP_UNFREEZE = suite in {"mtp-unfreeze", "mtp-parallel"}
    c.MTP_PARENT_ROOT = COMMON_ROOT / "mtp/runs/out"
    c.MTP_PARENT_INPUT_TOKENS = c.MTP_TRAIN_STEPS * c.MTP_STEP_TOKENS
    c.MTP_PARENT_STEP_TOKENS = c.MTP_STEP_TOKENS
    c.MTP_PARENT_CONTEXT = c.MTP_CONTEXT
    c.MTP_EMBED_LR = c.MTP_LR
    c.MTP_CHECKPOINT_FREE_BYTES = 20_000_000_000
    c.MTP_PRECHECK_PROMPTS = 2
    c.MTP_PRECHECK_NEW_TOKENS = 16
    if c.MTP_UNFREEZE:
        c.MTP_TRAIN_STEPS = (50_000_000 + c.MTP_STEP_TOKENS - 1) // c.MTP_STEP_TOKENS
        c.MTP_LR = 1e-5
        c.MTP_EMBED_LR = 1e-4
        c.MTP_WARMUP_STEPS = 30
        c.MTP_WEIGHT_DECAY = 0.01
        c.MTP_EVAL_INTERVAL = (10_000_000 + c.MTP_STEP_TOKENS - 1) // c.MTP_STEP_TOKENS
        c.MTP_EVAL_PROMPTS = 64
        c.MAX_HOURS = 4
    if suite in {"mtp", "mtp-unfreeze", "mtp-parallel"}:
        c.BLOCK_SIZE = c.MTP_CONTEXT
        c.GLOBAL_BATCH_SIZE = c.MTP_GLOBAL_BATCH
        c.TOKENS_PER_STEP = c.MTP_STEP_TOKENS
        c.TRAIN_TOKENS = c.MTP_TRAIN_STEPS * c.TOKENS_PER_STEP
        c.PROBE_STEPS = c.MTP_PROBE_STEPS
        c.experiment_configs = [
            ExperimentConfig(f"s_sps_mtp_n{n}_emb_1b", "s_sps_w0_20b", horizons=n)
            for n in (2, 3, 4)
        ]
        if c.MTP_UNFREEZE:
            c.experiment_configs = [
                ExperimentConfig(
                    f"s_sps_mtp_n{n}_unfreeze_ce_50m", "s_sps_w0_20b", horizons=n
                )
                for n in (2, 3, 4)
            ]

    # Parallel verifier numerical audit uses held-out prompts outside timed regions.
    c.MTP_PARALLEL_PROB_ATOL = 0.005
    c.MTP_PARALLEL_AUDIT_PROMPTS = 8

    c.MTP_PARALLEL_TIMEOUT_SECONDS = 4 * 3600

    return c
