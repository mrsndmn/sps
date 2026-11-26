"""SPS multi-token adaptation and verifier definitions."""

from dataclasses import dataclass, field
from pathlib import Path

from .environment import COMMON_ROOT
from .types import ExperimentConfig


@dataclass
class MTPSettings:
    MTP_SOURCE: Path = (
        COMMON_ROOT / "10b/runs/out/s_sps_w0_10b/ckpt_tokens_10003415040_final.pt"
    )
    MTP_DATA: Path = COMMON_ROOT / "10b/runs/data/fineweb-edu"
    MTP_CONTEXT: int = 1024
    MTP_MICRO_BATCH: int = 2
    MTP_GLOBAL_BATCH: int = 64
    MTP_STEP_TOKENS: int = MTP_CONTEXT * MTP_GLOBAL_BATCH
    MTP_TRAIN_STEPS: int = (1_000_000_000 + MTP_STEP_TOKENS - 1) // MTP_STEP_TOKENS
    MTP_PROBE_STEPS: int = 16
    MTP_LR: float = 0.01
    MTP_WARMUP_STEPS: int = 100
    MTP_WEIGHT_DECAY: float = 0.0
    MTP_BETAS: tuple[float, float] = (0.9, 0.95)
    MTP_GRAD_CLIP: float = 1.0
    MTP_EVAL_INTERVAL: int = 500
    MTP_VAL_TOKENS: int = 131072
    MTP_EVAL_PROMPTS: int = 16
    MTP_PROMPT_TOKENS: int = 128
    MTP_NEW_TOKENS: int = 128
    MTP_TEMPERATURE: float = 1.0
    MTP_SEED: int = 1337
    MTP_BACKEND: str = "auto"
    MTP_INFERENCE_BACKEND: str = "sdpa"
    MTP_MAX_HORIZONS: int = 4
    MTP_UNFREEZE_MAX_HOURS: int = 4
    MTP_UNFREEZE: bool = False
    MTP_PARENT_ROOT: Path = COMMON_ROOT / "mtp/runs/out"
    MTP_PARENT_INPUT_TOKENS: int = MTP_TRAIN_STEPS * MTP_STEP_TOKENS
    MTP_PARENT_STEP_TOKENS: int = MTP_STEP_TOKENS
    MTP_PARENT_CONTEXT: int = MTP_CONTEXT
    MTP_EMBED_LR: float = MTP_LR
    MTP_CHECKPOINT_FREE_BYTES: int = 20_000_000_000
    MTP_PRECHECK_PROMPTS: int = 2
    MTP_PRECHECK_NEW_TOKENS: int = 16
    MTP_PARALLEL_PROB_ATOL: float = 0.005
    MTP_PARALLEL_AUDIT_PROMPTS: int = 8
    MTP_PARALLEL_TIMEOUT_SECONDS: int = 4 * 3600
    experiment_configs: list[ExperimentConfig] = field(default_factory=list)


def configuration(suite: str) -> MTPSettings:
    c = MTPSettings(MTP_UNFREEZE=suite in {"mtp-unfreeze", "mtp-parallel"})
    if c.MTP_UNFREEZE:
        c.MTP_TRAIN_STEPS = (50_000_000 + c.MTP_STEP_TOKENS - 1) // c.MTP_STEP_TOKENS
        c.MTP_LR = 1e-5
        c.MTP_EMBED_LR = 1e-4
        c.MTP_WARMUP_STEPS = 30
        c.MTP_WEIGHT_DECAY = 0.01
        c.MTP_EVAL_INTERVAL = (10_000_000 + c.MTP_STEP_TOKENS - 1) // c.MTP_STEP_TOKENS
        c.MTP_EVAL_PROMPTS = 64
    if suite in {"mtp", "mtp-unfreeze", "mtp-parallel"}:
        suffix = "unfreeze_ce_50m" if c.MTP_UNFREEZE else "emb_1b"
        c.experiment_configs = [
            ExperimentConfig(f"s_sps_mtp_n{n}_{suffix}", "s_sps_w0_20b", horizons=n)
            for n in (2, 3, 4)
        ]
    return c
