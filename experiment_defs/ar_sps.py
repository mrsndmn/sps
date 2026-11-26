from dataclasses import dataclass
from types import SimpleNamespace


@dataclass(frozen=True)
class ARExperiment:
    name: str
    recipe: str
    num_gpus: int = 4
    trainable: bool = True
    distillation_weight: float = 0.0


def configuration(suite):
    c = SimpleNamespace()
    c.MODEL = "HuggingFaceTB/SmolLM2-135M"
    c.REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
    c.CONTEXT = 512
    c.GLOBAL_BATCH = 32
    c.MICRO_BATCH = 2
    c.SEED = 1337
    c.EMBED_LR = 1e-2
    c.FULL_LR = 1e-5
    c.WEIGHT_DECAY = 0.01
    c.BETAS = (0.9, 0.95)
    c.CLIP = 1.0
    c.PROBE_STEPS = 12
    c.WARMUP_PROBE = 3
    c.MAX_STEPS = 4096
    c.MIN_STEPS = 20
    c.STAGE_FRACTION = 0.2
    c.VAL_ROWS = 128
    c.TEST_ROWS = 256
    c.EVAL_BATCH = 4
    c.CHECKPOINT_PARTS = 4
    c.BOOTSTRAP_REPS = 2000
    c.CONFIDENCE = 0.95
    c.PRIMARY_METRICS = {
        "arc_easy": "acc_norm,none",
        "hellaswag": "acc_norm,none",
        "piqa": "acc_norm,none",
        "sciq": "acc_norm,none",
        "lambada_openai": "acc,none",
    }
    # Reservations include worker startup, all probes, validation and checkpoint I/O.
    c.GPU_HOURS = 96
    c.PREPARE_HOURS = 2  # four GPUs -> eight GPU-hours
    c.TRAIN_HOURS = 5  # three four-GPU workers -> sixty GPU-hours
    c.EVAL_HOURS = 4  # three training arms and two static baselines -> twenty GPU-hours
    c.SAFETY = 1.5
    c.STARTUP_SECONDS = 300
    c.NLL_ATOL = 0.0
    c.AUDIT_LENGTH = 32
    c.AUDIT_PREFIX = 16
    c.PARITY_ATOL = 2e-4
    c.PARITY_RTOL = 2e-4
    c.DATA_SPLIT_SALT = "ar-to-sps-v1"
    c.TRAIN_FRACTION = 0.98
    c.VAL_BOUNDARY = 0.99
    c.MAX_DATA_ROWS = c.MAX_STEPS * c.GLOBAL_BATCH

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
    c.DISTILLATION_TEMPERATURE = 1.0
    c.REFERENCE_SUITE = "ar-sps"
    c.REFERENCE_SIGNATURE = (
        "e9ed3e07768d5fb918301f5f23af9aa3020740a9578cb18f19543365a228a04d"
    )
    c.REFERENCE_DATA_SHA256 = {
        "train": "9d5f3fe5b6c0d88436145c49d721e821771fc5ee3c64988504deeb911a54796a",
        "val": "ee655e136153ce56db65e06fbb41711ebcbdb85fca338026d7c80e39b4c9df23",
        "test": "b227f003d9811440cb039991223eecbc8d4e7a7ab03108d7a7d1adb83f7274df",
    }
    if c.DISTILLATION:
        c.GPU_HOURS = 20
        c.TRAIN_HOURS = 2
        c.experiment_configs = [
            ARExperiment("smollm2_staged_sps_kd", "staged_sps", distillation_weight=0.5)
        ]
        assert c.PREPARE_HOURS * 4 + c.TRAIN_HOURS * 4 + c.EVAL_HOURS <= c.GPU_HOURS

    # Larger-model CE runs: separate data, artifacts, baselines and reservations.
    c.LARGE_CE_MODELS = {
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
    c.REFERENCE_CONTROLS = (
        "smollm2_staged_sps",
        "smollm2_direct_sps",
        "smollm2_continued_ar",
    )
    c.LARGE_KD_REFERENCES = {
        "ar-sps-360m-kd": (
            "ar-sps-360m-ce",
            "a91952db7534868c2d23050e447080d6682191ddb1df4ddd398c1db9bcbf8657",
        ),
        "ar-sps-1.7b-kd": (
            "ar-sps-1.7b-ce",
            "555846063de2a68b13a7efa818d5330be55fbf90074453ee5736ba4ad11df82a",
        ),
    }
    if suite in c.LARGE_KD_REFERENCES:
        c.DISTILLATION = True
        c.REFERENCE_SUITE, c.REFERENCE_SIGNATURE = c.LARGE_KD_REFERENCES[suite]
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

    c.REPORT_BASELINE = "smollm2_original_ar"
    c.ENDPOINT_COMPARISONS = (
        ("smollm2_direct_sps", "smollm2_staged_sps"),
        ("smollm2_continued_ar", "smollm2_staged_sps"),
    )
    c.QWEN_MODELS = {
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
