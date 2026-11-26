"""Step-384 evaluation and otherwise matched warmup/cosine instruct reruns."""
SOURCE_SUITE = 'ar-sps-qwen3-1.7b-instruct'
SOURCE_SIGNATURE = '761505551994ceec0a907401b3ee875ffea744b69bc94c7007b23c58181b466b'
ROOT_NAMES = {'step384': 'qwen3-instruct-step384', 'cosine': 'qwen3-instruct-cosine'}
EVAL_STEP = 384
TRAIN_STEPS = 512
STAGE_STEPS = 102
WARMUP_FRACTION = .3
MIN_LR_RATIO = 0.
SCHEDULE_SCOPE = 'per_stage'
TRAIN_HOURS = 8
WORKLOAD_HOURS = {'quality': 4, 'ifeval-nonthinking': 12, 'ifeval-thinking': 12,
                  'mmlu-nonthinking': 72, 'mmlu-thinking': 72}
WORKLOADS = tuple(WORKLOAD_HOURS)
STARTUP_SECONDS = 300
BUDGETS = {'step384': 4*sum(WORKLOAD_HOURS.values()),
           'cosine': 4*4*TRAIN_HOURS + 4*sum(WORKLOAD_HOURS.values())}
