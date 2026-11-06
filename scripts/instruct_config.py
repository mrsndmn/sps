"""Isolated, bounded Qwen3 instruct adaptation experiment design."""
from dataclasses import dataclass

SUITE = 'ar-sps-qwen3-1.7b-instruct'
MODEL = 'Qwen/Qwen3-1.7B'
REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
SEED = 1337
CONTEXT = 2048
GLOBAL_BATCH = 16
MICRO_BATCH = 1
MAX_STEPS = 512
MIN_STEPS = 32
STAGE_FRACTION = .2
PROBE_STEPS = 8
WARMUP_PROBE = 1
EMBED_LR = 1e-2
FULL_LR = 1e-5
BETAS = (.9, .95)
WEIGHT_DECAY = .01
CLIP = 1.
CHECKPOINT_PARTS = 4
DISTILLATION_TEMPERATURE = 1.
DATASET_REPO = 'HuggingFaceTB/smoltalk'
DATASET_REVISION = '5feaf2fd3ffca7c237fc38d1861bc30365d48ffa'
DATA_FILES = tuple(f'data/all/train-{i:05d}-of-00009.parquet' for i in range(9))
DATA_ROWS = {'train': MAX_STEPS*GLOBAL_BATCH, 'val': 128, 'test': 128}
DATA_SPLIT_SALT = 'qwen3-instruct-dialog-v1'
TRAIN_FRACTION = .96
VAL_BOUNDARY = .98
THINKING_FRACTION = .5
DATA_CANDIDATE_MULTIPLIER = 2
MIN_DATA_FRACTION = .5
TEACHER_MAX_NEW_TOKENS = 1024
TEACHER_TEMPERATURE = .6
TEACHER_TOP_P = .95
TEACHER_TOP_K = 20
MMLU_DATASET = 'cais/mmlu'
MMLU_REVISION = 'c30699e8356da336a370243923dbaf21066bb9fe'
IFEVAL_DATASET = 'google/IFEval'
IFEVAL_REVISION = '966cd89545d6b6acfd7638bc708b98261ca58e84'
CONTAMINATION_DATASETS = (
    dict(repo=MMLU_DATASET, config='all', revision=MMLU_REVISION, split='test', fields=('question',)),
    dict(repo=IFEVAL_DATASET, revision=IFEVAL_REVISION, split='train', fields=('prompt',)),
    dict(repo='allenai/ai2_arc', config='ARC-Easy', revision='210d026faf9955653af8916fad021475a3f00453', split='test', fields=('question',)),
    dict(repo='Rowan/hellaswag', revision='218ec52e09a7e7462a5400043bb9a69a41d06b76', split='validation', fields=('ctx',)),
    dict(repo='baber/piqa', revision='142f6d7367fd9877f0fb3b5734ea6a545f54cdd1', split='validation', fields=('goal',)),
    dict(repo='allenai/sciq', revision='2c94ad3e1aafab77146f384e23536f97a4849815', split='test', fields=('question',)),
    dict(repo='EleutherAI/lambada_openai', config='default', revision='900124bf3b8235c6daf21033af9948b3f07346c4', split='test', fields=('text',), trust_remote_code=True),
)
GENERATION = {
    'thinking': dict(temperature=.6, top_p=.95, top_k=20, max_new_tokens=2048),
    'nonthinking': dict(temperature=.7, top_p=.8, top_k=20, max_new_tokens=512),
}
GENERATION_AUDIT_COUNT = 8
AUDIT_LENGTH = 32
GENERATION_PARITY_ATOL = 2e-4
GENERATION_PARITY_RTOL = 2e-4
LIKELIHOOD_CONTEXT = 512
LIKELIHOOD_TASKS = ('arc_easy', 'hellaswag', 'piqa', 'sciq', 'lambada_openai')
LIKELIHOOD_DATASETS = dict(zip(LIKELIHOOD_TASKS, CONTAMINATION_DATASETS[2:]))
EVAL_BATCH = 1
# Queue waiting is not charged; these are bounded worker execution reservations.
PREPARE_HOURS = 24
TRAIN_HOURS = 8
EVAL_HOURS = 72
EVAL_MAX_HOURS = EVAL_HOURS - 1
STARTUP_SECONDS = 300
SAFETY = 2.

@dataclass(frozen=True)
class InstructExperiment:
    name: str
    recipe: str
    num_gpus: int = 4
    trainable: bool = True
    distillation_weight: float = 0.

experiment_configs = [
    InstructExperiment('qwen3_instruct_original_ar', 'original_ar', trainable=False),
    InstructExperiment('qwen3_instruct_staged_sps_ce', 'staged_sps'),
    InstructExperiment('qwen3_instruct_staged_sps_kd', 'staged_sps', distillation_weight=.5),
    InstructExperiment('qwen3_instruct_continued_ar_ce', 'continued_ar'),
    InstructExperiment('qwen3_instruct_continued_ar_kd', 'continued_ar', distillation_weight=.5),
]
GPU_HOURS = PREPARE_HOURS*4 + TRAIN_HOURS*4*4 + EVAL_HOURS*len(experiment_configs)
