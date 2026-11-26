"""Executable design for the bounded pretrained AR-to-SPS pilot."""

from dataclasses import dataclass, field

from .types import ARExperiment


@dataclass
class ARSettings:
    MODEL: str = "HuggingFaceTB/SmolLM2-135M"
    REVISION: str = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
    CONTEXT: int = 512
    GLOBAL_BATCH: int = 32
    MICRO_BATCH: int = 2
    SEED: int = 1337
    EMBED_LR: float = 1e-2
    FULL_LR: float = 1e-5
    WEIGHT_DECAY: float = 0.01
    BETAS: tuple[float, float] = (0.9, 0.95)
    CLIP: float = 1.0
    PROBE_STEPS: int = 12
    WARMUP_PROBE: int = 3
    MAX_STEPS: int = 4096
    MIN_STEPS: int = 20
    STAGE_FRACTION: float = 0.2
    VAL_ROWS: int = 128
    TEST_ROWS: int = 256
    EVAL_BATCH: int = 4
    CHECKPOINT_PARTS: int = 4
    BOOTSTRAP_REPS: int = 2000
    CONFIDENCE: float = 0.95
    PRIMARY_METRICS: dict[str, str] = field(
        default_factory=lambda: {
            "arc_easy": "acc_norm,none",
            "hellaswag": "acc_norm,none",
            "piqa": "acc_norm,none",
            "sciq": "acc_norm,none",
            "lambada_openai": "acc,none",
        }
    )
    GPU_HOURS: int = 96
    PREPARE_HOURS: int = 2  # four GPUs -> eight GPU-hours
    TRAIN_HOURS: int = 5  # three four-GPU workers -> sixty GPU-hours
    EVAL_HOURS: int = (
        4  # three training arms and two static baselines -> twenty GPU-hours
    )
    SAFETY: float = 1.5
    STARTUP_SECONDS: int = 300
    NLL_ATOL: float = 0.0
    AUDIT_LENGTH: int = 32
    AUDIT_PREFIX: int = 16
    PARITY_ATOL: float = 2e-4
    PARITY_RTOL: float = 2e-4
    DATA_SPLIT_SALT: str = "ar-to-sps-v1"
    TRAIN_FRACTION: float = 0.98
    VAL_BOUNDARY: float = 0.99
    DISTILLATION_TEMPERATURE: float = 1.0
    REFERENCE_SUITE: str = "ar-sps"
    REFERENCE_DATA_SHA256: dict[str, str] = field(
        default_factory=lambda: {
            "train": "9d5f3fe5b6c0d88436145c49d721e821771fc5ee3c64988504deeb911a54796a",
            "val": "ee655e136153ce56db65e06fbb41711ebcbdb85fca338026d7c80e39b4c9df23",
            "test": "b227f003d9811440cb039991223eecbc8d4e7a7ab03108d7a7d1adb83f7274df",
        }
    )
    LARGE_CE_MODELS: dict[str, tuple[str, str, int, int, int]] = field(
        default_factory=lambda: {
            "ar-sps-360m-ce": (
                "HuggingFaceTB/SmolLM2-360M",
                "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
                2,
                4,
                4,
            ),
            "ar-sps-1.7b-ce": (
                "HuggingFaceTB/SmolLM2-1.7B",
                "effd688a12921b4cc83e3312b6feb579f70f9c71",
                1,
                12,
                12,
            ),
        }
    )
    REFERENCE_CONTROLS: tuple[str, ...] = (
        "smollm2_staged_sps",
        "smollm2_direct_sps",
        "smollm2_continued_ar",
    )
    LARGE_KD_REFERENCES: dict[str, str] = field(
        default_factory=lambda: {
            "ar-sps-360m-kd": "ar-sps-360m-ce",
            "ar-sps-1.7b-kd": "ar-sps-1.7b-ce",
        }
    )
    REPORT_BASELINE: str = "smollm2_original_ar"
    ENDPOINT_COMPARISONS: tuple[tuple[str, str], ...] = (
        ("smollm2_direct_sps", "smollm2_staged_sps"),
        ("smollm2_continued_ar", "smollm2_staged_sps"),
    )
    QWEN_MODELS: dict[str, tuple[str, str, int, int]] = field(
        default_factory=lambda: {
            "ar-sps-qwen3-0.6b": (
                "Qwen/Qwen3-0.6B-Base",
                "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
                4,
                4,
            ),
            "ar-sps-qwen3-1.7b": (
                "Qwen/Qwen3-1.7B-Base",
                "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
                6,
                8,
            ),
        }
    )
    DISTILLATION: bool = False
    LARGE_CE: bool = False
    QWEN: bool = False
    experiment_configs: list[ARExperiment] = field(default_factory=list)

    @property
    def MAX_DATA_ROWS(self):
        return self.MAX_STEPS * self.GLOBAL_BATCH


def configuration(suite: str) -> ARSettings:
    c = ARSettings()

    c.experiment_configs = [
        ARExperiment("smollm2_original_ar", "original_ar", trainable=False),
        ARExperiment("smollm2_untrained_sps", "untrained_sps", trainable=False),
        ARExperiment("smollm2_staged_sps", "staged_sps"),
        ARExperiment("smollm2_direct_sps", "direct_sps"),
        ARExperiment("smollm2_continued_ar", "continued_ar"),
    ]
    assert (
        c.PREPARE_HOURS * 4
        + c.TRAIN_HOURS * 4 * 3
        + c.EVAL_HOURS * len(c.experiment_configs)
        <= c.GPU_HOURS
    )

    # Follow-up: a fresh staged student, matched to the completed CE pilot.
    c.DISTILLATION = suite == "ar-sps-kd"
    if c.DISTILLATION:
        c.GPU_HOURS = 20
        c.TRAIN_HOURS = 2
        c.experiment_configs = [
            ARExperiment("smollm2_staged_sps_kd", "staged_sps", distillation_weight=0.5)
        ]
        assert c.PREPARE_HOURS * 4 + c.TRAIN_HOURS * 4 + c.EVAL_HOURS <= c.GPU_HOURS

    # Larger-model CE runs: separate data, artifacts, baselines and reservations.
    c.LARGE_CE = suite in c.LARGE_CE_MODELS
    if c.LARGE_CE:
        c.MODEL, c.REVISION, c.MICRO_BATCH, c.TRAIN_HOURS, c.EVAL_HOURS = (
            c.LARGE_CE_MODELS[suite]
        )
        c.EVAL_BATCH = c.MICRO_BATCH
        c.experiment_configs = [
            ARExperiment("smollm2_original_ar", "original_ar", trainable=False),
            ARExperiment("smollm2_staged_sps", "staged_sps"),
        ]
        c.GPU_HOURS = (
            c.PREPARE_HOURS * 4
            + c.TRAIN_HOURS * 4
            + c.EVAL_HOURS * len(c.experiment_configs)
        )

    # Same-size frozen source teachers; fresh students matched to each completed CE run.
    if suite in c.LARGE_KD_REFERENCES:
        c.DISTILLATION = True
        c.REFERENCE_SUITE = c.LARGE_KD_REFERENCES[suite]
        c.MODEL, c.REVISION, c.MICRO_BATCH, c.TRAIN_HOURS, c.EVAL_HOURS = (
            c.LARGE_CE_MODELS[c.REFERENCE_SUITE]
        )
        c.EVAL_BATCH = c.MICRO_BATCH
        # All three pinned CE tokenized corpora have identical verified split hashes.
        c.REFERENCE_CONTROLS = ("smollm2_staged_sps",)
        c.experiment_configs = [
            ARExperiment("smollm2_staged_sps_kd", "staged_sps", distillation_weight=0.5)
        ]
        c.GPU_HOURS = c.PREPARE_HOURS * 4 + c.TRAIN_HOURS * 4 + c.EVAL_HOURS

    c.QWEN = suite in c.QWEN_MODELS
    if c.QWEN:
        c.MODEL, c.REVISION, c.TRAIN_HOURS, c.EVAL_HOURS = c.QWEN_MODELS[suite]
        c.MICRO_BATCH = c.EVAL_BATCH = (
            1  # larger native vocabulary, including full KL logits
        )
        c.REPORT_BASELINE = "qwen3_original_ar"
        c.ENDPOINT_COMPARISONS = (("qwen3_staged_sps_ce", "qwen3_staged_sps_kd"),)
        # DISTILLATION selects historical-reference reuse, not the per-arm objective.
        # Both fresh Qwen arms share a newly prepared native-tokenizer corpus.
        c.experiment_configs = [
            ARExperiment(c.REPORT_BASELINE, "original_ar", trainable=False),
            ARExperiment("qwen3_staged_sps_ce", "staged_sps"),
            ARExperiment("qwen3_staged_sps_kd", "staged_sps", distillation_weight=0.5),
        ]
        c.GPU_HOURS = c.PREPARE_HOURS * 4 + c.TRAIN_HOURS * 4 * 2 + c.EVAL_HOURS * 3

    return c
