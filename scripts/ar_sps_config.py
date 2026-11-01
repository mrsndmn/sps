"""Executable design for the bounded pretrained AR-to-SPS pilot."""
from dataclasses import dataclass
import os

MODEL = 'HuggingFaceTB/SmolLM2-135M'
REVISION = '93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
CONTEXT = 512
GLOBAL_BATCH = 32
MICRO_BATCH = 2
SEED = 1337
EMBED_LR = 1e-2
FULL_LR = 1e-5
WEIGHT_DECAY = .01
BETAS = (.9, .95)
CLIP = 1.
PROBE_STEPS = 12
WARMUP_PROBE = 3
MAX_STEPS = 4096
MIN_STEPS = 20
STAGE_FRACTION = .2
VAL_ROWS = 128
TEST_ROWS = 256
EVAL_BATCH = 4
CHECKPOINT_PARTS = 4
BOOTSTRAP_REPS = 2000
CONFIDENCE = .95
PRIMARY_METRICS = {'arc_easy': 'acc_norm,none', 'hellaswag': 'acc_norm,none',
                   'piqa': 'acc_norm,none', 'sciq': 'acc_norm,none', 'lambada_openai': 'acc,none'}
# Reservations include worker startup, all probes, validation and checkpoint I/O.
GPU_HOURS = 96
PREPARE_HOURS = 2  # four GPUs -> eight GPU-hours
TRAIN_HOURS = 5  # three four-GPU workers -> sixty GPU-hours
EVAL_HOURS = 4  # three training arms and two static baselines -> twenty GPU-hours
SAFETY = 1.5
STARTUP_SECONDS = 300
NLL_ATOL = 0.0
AUDIT_LENGTH = 32
AUDIT_PREFIX = 16
PARITY_ATOL = 2e-4
PARITY_RTOL = 2e-4
DATA_SPLIT_SALT = 'ar-to-sps-v1'
TRAIN_FRACTION = .98
VAL_BOUNDARY = .99
MAX_DATA_ROWS = MAX_STEPS * GLOBAL_BATCH


@dataclass(frozen=True)
class ARExperiment:
    name: str
    recipe: str
    num_gpus: int = 4
    trainable: bool = True
    distillation_weight: float = 0.0


experiment_configs = [
    ARExperiment('smollm2_original_ar', 'original_ar', trainable=False),
    ARExperiment('smollm2_untrained_sps', 'untrained_sps', trainable=False),
    ARExperiment('smollm2_staged_sps', 'staged_sps'),
    ARExperiment('smollm2_direct_sps', 'direct_sps'),
    ARExperiment('smollm2_continued_ar', 'continued_ar'),
]
assert PREPARE_HOURS*4 + TRAIN_HOURS*4*3 + EVAL_HOURS*len(experiment_configs) <= GPU_HOURS


# Follow-up: a fresh staged student, matched to the completed CE pilot.
DISTILLATION = os.environ.get('SPS_EXPERIMENT_SUITE') == 'ar-sps-kd'
DISTILLATION_TEMPERATURE = 1.0
REFERENCE_SUITE = 'ar-sps'
REFERENCE_SIGNATURE = 'e9ed3e07768d5fb918301f5f23af9aa3020740a9578cb18f19543365a228a04d'
REFERENCE_DATA_SHA256 = {
    'train': '9d5f3fe5b6c0d88436145c49d721e821771fc5ee3c64988504deeb911a54796a',
    'val': 'ee655e136153ce56db65e06fbb41711ebcbdb85fca338026d7c80e39b4c9df23',
    'test': 'b227f003d9811440cb039991223eecbc8d4e7a7ab03108d7a7d1adb83f7274df',
}
if DISTILLATION:
    GPU_HOURS = 20
    TRAIN_HOURS = 2
    experiment_configs = [ARExperiment('smollm2_staged_sps_kd', 'staged_sps', distillation_weight=.5)]
    assert PREPARE_HOURS*4 + TRAIN_HOURS*4 + EVAL_HOURS <= GPU_HOURS


# Larger-model CE runs: separate data, artifacts, baselines and reservations.
LARGE_CE_MODELS = {
    'ar-sps-360m-ce': ('HuggingFaceTB/SmolLM2-360M',
                      'f8027fd0eaeea54caa13c31d31b9fdc459c38b49', 2, 4, 4),
    'ar-sps-1.7b-ce': ('HuggingFaceTB/SmolLM2-1.7B',
                      'effd688a12921b4cc83e3312b6feb579f70f9c71', 1, 12, 12),
}
LARGE_CE = os.environ.get('SPS_EXPERIMENT_SUITE') in LARGE_CE_MODELS
if LARGE_CE:
    MODEL, REVISION, MICRO_BATCH, TRAIN_HOURS, EVAL_HOURS = LARGE_CE_MODELS[os.environ['SPS_EXPERIMENT_SUITE']]
    EVAL_BATCH = MICRO_BATCH
    experiment_configs = [
        ARExperiment('smollm2_original_ar', 'original_ar', trainable=False),
        ARExperiment('smollm2_staged_sps', 'staged_sps'),
    ]
    GPU_HOURS = PREPARE_HOURS*4 + TRAIN_HOURS*4 + EVAL_HOURS*len(experiment_configs)


# Same-size frozen source teachers; fresh students matched to each completed CE run.
REFERENCE_CONTROLS = ('smollm2_staged_sps', 'smollm2_direct_sps', 'smollm2_continued_ar')
LARGE_KD_REFERENCES = {
    'ar-sps-360m-kd': ('ar-sps-360m-ce',
                       'a91952db7534868c2d23050e447080d6682191ddb1df4ddd398c1db9bcbf8657'),
    'ar-sps-1.7b-kd': ('ar-sps-1.7b-ce',
                       '555846063de2a68b13a7efa818d5330be55fbf90074453ee5736ba4ad11df82a'),
}
if os.environ.get('SPS_EXPERIMENT_SUITE') in LARGE_KD_REFERENCES:
    DISTILLATION = True
    REFERENCE_SUITE, REFERENCE_SIGNATURE = LARGE_KD_REFERENCES[os.environ['SPS_EXPERIMENT_SUITE']]
    MODEL, REVISION, MICRO_BATCH, TRAIN_HOURS, EVAL_HOURS = LARGE_CE_MODELS[REFERENCE_SUITE]
    EVAL_BATCH = MICRO_BATCH
    # All three pinned CE tokenized corpora have identical verified split hashes.
    REFERENCE_CONTROLS = ('smollm2_staged_sps',)
    experiment_configs = [ARExperiment('smollm2_staged_sps_kd', 'staged_sps', distillation_weight=.5)]
    GPU_HOURS = PREPARE_HOURS*4 + TRAIN_HOURS*4 + EVAL_HOURS


REPORT_BASELINE = 'smollm2_original_ar'
ENDPOINT_COMPARISONS = (('smollm2_direct_sps', 'smollm2_staged_sps'),
                        ('smollm2_continued_ar', 'smollm2_staged_sps'))
QWEN_MODELS = {
    'ar-sps-qwen3-0.6b': ('Qwen/Qwen3-0.6B-Base',
                         'da87bfb608c14b7cf20ba1ce41287e8de496c0cd', 4, 4),
    'ar-sps-qwen3-1.7b': ('Qwen/Qwen3-1.7B-Base',
                         'ea980cb0a6c2ae4b936e82123acc929f1cec04c1', 6, 8),
}
QWEN = os.environ.get('SPS_EXPERIMENT_SUITE') in QWEN_MODELS
if QWEN:
    MODEL, REVISION, TRAIN_HOURS, EVAL_HOURS = QWEN_MODELS[os.environ['SPS_EXPERIMENT_SUITE']]
    MICRO_BATCH = EVAL_BATCH = 1  # larger native vocabulary, including full KL logits
    REPORT_BASELINE = 'qwen3_original_ar'
    ENDPOINT_COMPARISONS = (('qwen3_staged_sps_ce', 'qwen3_staged_sps_kd'),)
    # DISTILLATION selects historical-reference reuse, not the per-arm objective.
    # Both fresh Qwen arms share a newly prepared native-tokenizer corpus.
    experiment_configs = [
        ARExperiment(REPORT_BASELINE, 'original_ar', trainable=False),
        ARExperiment('qwen3_staged_sps_ce', 'staged_sps'),
        ARExperiment('qwen3_staged_sps_kd', 'staged_sps', distillation_weight=.5),
    ]
    GPU_HOURS = PREPARE_HOURS*4 + TRAIN_HOURS*4*2 + EVAL_HOURS*3
