"""Continue constant-LR instruct arms on the same corpus for four epochs."""
SOURCE_SUITE = 'ar-sps-qwen3-1.7b-instruct'
SOURCE_SIGNATURE = '761505551994ceec0a907401b3ee875ffea744b69bc94c7007b23c58181b466b'
START_STEP = 512
TRAIN_STEPS = 2048
EVAL_STEPS = (1024, 1536, 2048)
ROOT_NAME = 'qwen3-instruct-extended-2048'
ROOT_NAMES = {str(step): ROOT_NAME for step in EVAL_STEPS}
CHECKPOINT_INTERVAL = 128
TRAIN_HOURS = 8
WORKLOAD_HOURS = {'ifeval-nonthinking': 12, 'ifeval-thinking': 12}
WORKLOADS = tuple(WORKLOAD_HOURS)
STARTUP_SECONDS = 300
BUDGET = 4*4*TRAIN_HOURS + 4*len(EVAL_STEPS)*sum(WORKLOAD_HOURS.values())
BUDGETS = {key: BUDGET for key in ROOT_NAMES}

import random
from functools import lru_cache

@lru_cache(maxsize=4)
def epoch_order(count,epoch,seed):
    order=list(range(count))
    if epoch: random.Random(seed+epoch).shuffle(order)
    return tuple(order)


def global_batch_indices(step,count,batch_size,seed):
    if count % batch_size or count < batch_size or step < 0:
        raise ValueError('Dataset must contain full global batches')
    epoch,offset=divmod(step*batch_size,count)
    return epoch_order(count,epoch,seed)[offset:offset+batch_size]
