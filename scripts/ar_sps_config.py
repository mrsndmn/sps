"""Executable design for the bounded pretrained AR-to-SPS pilot."""
from dataclasses import dataclass

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


experiment_configs = [
    ARExperiment('smollm2_original_ar', 'original_ar', trainable=False),
    ARExperiment('smollm2_untrained_sps', 'untrained_sps', trainable=False),
    ARExperiment('smollm2_staged_sps', 'staged_sps'),
    ARExperiment('smollm2_direct_sps', 'direct_sps'),
    ARExperiment('smollm2_continued_ar', 'continued_ar'),
]
assert PREPARE_HOURS*4 + TRAIN_HOURS*4*3 + EVAL_HOURS*len(experiment_configs) <= GPU_HOURS
