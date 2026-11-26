"""Medusa, tiny AR and history-lookup baseline definitions."""

from dataclasses import dataclass, field

from .types import ExperimentConfig


@dataclass(kw_only=True)
class FamilyConfig:
    BASELINE_CONTEXT: int = 1024
    BASELINE_GLOBAL_BATCH: int = 64
    BASELINE_MICRO_BATCH: int = 2
    BASELINE_STEPS: int = 15259 + 763
    BASELINE_PROBE_STEPS: int = 16
    BASELINE_RESERVE_HORIZONS: int = 5
    BASELINE_MEDUSA_HEADS: int = 4
    BASELINE_LR: dict[str, float] = field(
        default_factory=lambda: {"medusa": 1e-3, "ar_tiny": 3e-4}
    )
    BASELINE_WARMUP: int = 100
    BASELINE_WEIGHT_DECAY: float = 0.01
    BASELINE_BETAS: tuple[float, ...] = (0.9, 0.95)
    BASELINE_CLIP: float = 1.0
    BASELINE_EVAL_INTERVAL: int = 500
    BASELINE_VAL_TOKENS: int = 131072
    BASELINE_MAX_HOURS: int = 24
    BASELINE_AR_ARGS: dict[str, int] = field(
        default_factory=lambda: dict(
            n_layer=4, n_head=4, hidden_size=256, intermediate_size=768
        )
    )
    BASELINE_LOOKUP_MATCH: int = 8
    BASELINE_EVAL_HORIZONS: tuple[int, ...] = (2, 3, 4)
    BASELINE_EVAL_PROMPTS: int = 64
    BASELINE_PROMPT_TOKENS: int = 128
    BASELINE_NEW_TOKENS: int = 128
    BASELINE_WARMUP_PASSES: int = 2
    BASELINE_EVAL_TIMEOUT: int = 4 * 3600

    @property
    def BASELINE_STEP_TOKENS(self) -> int:
        return self.BASELINE_CONTEXT * self.BASELINE_GLOBAL_BATCH


family_config = FamilyConfig()
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
