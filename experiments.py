"""Source of truth for the shortened, token-matched S-scale comparison."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

ARTIFACTS_ROOT = Path('/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/sps-artifacts')
RUN_ROOT = ARTIFACTS_ROOT / 'runs'
SANITY_ROOT = ARTIFACTS_ROOT / 'sanity'
STAGING_ROOT = ARTIFACTS_ROOT / 'code'
HF_HOME = ARTIFACTS_ROOT / 'hf'
PYTHON = Path('/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/sps/.venv/bin/python')
INSTANCE_TYPES_BY_NUM_GPUS = {4: 'a100.4gpu'}
BASE_IMAGE = 'cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41'
DATASET_REPO = 'HuggingFaceFW/fineweb-edu'
DATASET_REVISION = '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
# Three complete sample-10BT shards supply roughly two billion source tokens.
DATA_FILES = tuple(f'sample/10BT/{i:03d}_00000.parquet' for i in range(3))
EXISTING_HF_CACHES = (
    Path('/home/jovyan/.cache/huggingface/hub'),
    Path('/mnt/virtual_ai0001053-00054_SR004-nfs2/.cache/huggingface/hub'),
    Path('/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/hf-cache/hub'),
    Path('/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/HF_CACHE_NFS1/hub'),
)
TASKS = ('arc_easy', 'hellaswag', 'piqa', 'sciq', 'lambada_openai')
GLOBAL_BATCH_SIZE = 96
BLOCK_SIZE = 4096
TOKENS_PER_STEP = GLOBAL_BATCH_SIZE * BLOCK_SIZE
TRAIN_TOKENS = 2544 * TOKENS_PER_STEP  # 1,000,341,504; exact update boundary
PROBE_STEPS = 16
MAX_HOURS = 24
VAL_FRACTION = 0.01
VAL_SEED = 2357


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    recipe: str
    num_gpus: int = 4


experiment_configs = [
    ExperimentConfig('s_standard_pilot_1b', 's_full_attention_20b'),
    ExperimentConfig('s_sps_w0_pilot_1b', 's_sps_w0_20b'),
]


def collect_experiments(filter_name=None):
    return [e for e in experiment_configs if filter_name is None or filter_name in e.name]


def overrides(exp, *, probe=False):
    root = SANITY_ROOT if probe else RUN_ROOT
    return [
        f'+experiment={exp.recipe}', f'system.data_root={root}',
        f'experiment.name={exp.name}', f'logging.wandb_run_name={exp.name}',
        'logging.wandb_log=false', 'system.compile=false',
        'training.micro_batch_size=6', f'training.global_batch_size={GLOBAL_BATCH_SIZE}',
        f'training.max_tokens={PROBE_STEPS * TOKENS_PER_STEP if probe else TRAIN_TOKENS}',
        f'training.eval_total_tokens={TOKENS_PER_STEP if probe else 1048576}',
        f'training.eval_interval_tokens={TRAIN_TOKENS if probe else 50000000}',
        f'training.skip_first_eval={str(probe).lower()}',
        f'training.init_from={"scratch" if probe else "resume"}',
        f'training.save_every={TRAIN_TOKENS if probe else 250000000}',
        f'training.rolling_save_every={TRAIN_TOKENS if probe else 100000000}',
        f'scheduler.warmup_tokens={TOKENS_PER_STEP if probe else 10000000}',
        f'scheduler.lr_decay_tokens={TOKENS_PER_STEP if probe else 100000000}',
    ]


def train_command(exp, *, probe=False):
    return [str(PYTHON), '-m', 'torch.distributed.run', '--standalone',
            f'--nproc_per_node={exp.num_gpus}', 'scripts/train.py', *overrides(exp, probe=probe)]


def eval_command(exp, task):
    return [str(PYTHON), 'scripts/evaluate.py', *overrides(exp), '+checkpoint=final',
            f'+tasks={task}', '+eval_batch_size=6', '+num_fewshot=0']


def output_dir(exp):
    return RUN_ROOT / 'out' / exp.name


def design_signature():
    """Reject stale completion/probe markers after a design change."""
    design = dict(repo=DATASET_REPO, revision=DATASET_REVISION, files=DATA_FILES,
                  val_fraction=VAL_FRACTION, val_seed=VAL_SEED,
                  commands=[train_command(e) for e in experiment_configs],
                  probes=[train_command(e, probe=True) for e in experiment_configs], tasks=TASKS)
    return hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
