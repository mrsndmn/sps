"""Source of truth for the shortened, token-matched S-scale comparison."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os

SUITE = os.environ.get('SPS_EXPERIMENT_SUITE', '1b')
if SUITE not in {'1b', '10b', 'mtp', 'mtp-unfreeze', 'mtp-parallel', 'spec-baselines', 'mtp-unified'}:
    raise ValueError(f'Unknown SPS_EXPERIMENT_SUITE={SUITE!r}')
COMMON_ROOT = Path('/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/sps-artifacts')
ARTIFACTS_ROOT = COMMON_ROOT if SUITE == '1b' else COMMON_ROOT / SUITE
RUN_ROOT = ARTIFACTS_ROOT / 'runs'
SANITY_ROOT = ARTIFACTS_ROOT / 'sanity'
STAGING_ROOT = COMMON_ROOT / 'code'
HF_HOME = COMMON_ROOT / 'hf'
PYTHON = Path('/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/sps/.venv/bin/python')
INSTANCE_TYPES_BY_NUM_GPUS = {4: 'a100.4gpu'}
BASE_IMAGE = 'cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41'
DATASET_REPO = 'HuggingFaceFW/fineweb-edu'
DATASET_REVISION = '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
# Three complete sample-10BT shards supply roughly two billion source tokens.
DATA_FILES = tuple(f'sample/10BT/{i:03d}_00000.parquet' for i in range(3))
if SUITE == '10b':
    # Spread the larger corpus across crawl partitions, with >10B GPT-2 tokens
    # after the held-out split. Reuse the same pinned repository/cache.
    DATA_FILES = (tuple(f'sample/100BT/{i:03d}_00000.parquet' for i in range(14))
                  + ('sample/100BT/000_00001.parquet', 'sample/100BT/001_00001.parquet'))
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
CHUNK_SIZE_UNITS = 1024
TRAINING_SEED = 1337
TRAIN_TOKENS = 2544 * TOKENS_PER_STEP  # 1,000,341,504; exact update boundary
if SUITE == '10b':
    TRAIN_TOKENS *= 10
PROBE_STEPS = 16
MAX_HOURS = 24
VAL_FRACTION = 0.01
VAL_SEED = 2357
MIN_FREE_BYTES = (120 if SUITE == '10b' else 40) * 10**9
# The long run has a complete prior run for calibration rather than only a
# short probe. Keep a margin plus one hour for startup and checkpoint I/O.
RUNTIME_SAFETY_FACTOR = 1.15 if SUITE == '10b' else 1.5


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    recipe: str
    num_gpus: int = 4
    reference_name: str | None = None
    horizons: int = 1
    trainable: bool = True


experiment_configs = [
    ExperimentConfig('s_standard_pilot_1b', 's_full_attention_20b'),
    ExperimentConfig('s_sps_w0_pilot_1b', 's_sps_w0_20b'),
]
if SUITE == '10b':
    experiment_configs = [
        ExperimentConfig('s_standard_10b', 's_full_attention_20b',
                         reference_name='s_standard_pilot_1b'),
        ExperimentConfig('s_sps_w0_10b', 's_sps_w0_20b',
                         reference_name='s_sps_w0_pilot_1b'),
    ]


# Embedding-only adaptation: source settings and benchmark workloads live here.
MTP_SOURCE = COMMON_ROOT / '10b/runs/out/s_sps_w0_10b/ckpt_tokens_10003415040_final.pt'
MTP_DATA = COMMON_ROOT / '10b/runs/data/fineweb-edu'
MTP_CONTEXT = 1024
MTP_MICRO_BATCH = 2
MTP_GLOBAL_BATCH = 64
MTP_STEP_TOKENS = MTP_CONTEXT * MTP_GLOBAL_BATCH
MTP_TRAIN_STEPS = (1_000_000_000 + MTP_STEP_TOKENS - 1) // MTP_STEP_TOKENS
MTP_PROBE_STEPS = 16
MTP_LR = 0.01
MTP_WARMUP_STEPS = 100
MTP_WEIGHT_DECAY = 0.0
MTP_BETAS = (0.9, 0.95)
MTP_GRAD_CLIP = 1.0
MTP_EVAL_INTERVAL = 500
MTP_VAL_TOKENS = 131072
MTP_EVAL_PROMPTS = 16
MTP_PROMPT_TOKENS = 128
MTP_NEW_TOKENS = 128
MTP_TEMPERATURE = 1.0
MTP_SEED = 1337
MTP_BACKEND = 'auto'
MTP_INFERENCE_BACKEND = 'sdpa'
MTP_MAX_HORIZONS = 4
MTP_UNFREEZE = SUITE in {'mtp-unfreeze', 'mtp-parallel'}
MTP_PARENT_ROOT = COMMON_ROOT / 'mtp/runs/out'
MTP_PARENT_INPUT_TOKENS = MTP_TRAIN_STEPS * MTP_STEP_TOKENS
MTP_PARENT_STEP_TOKENS = MTP_STEP_TOKENS
MTP_PARENT_CONTEXT = MTP_CONTEXT
MTP_EMBED_LR = MTP_LR
MTP_CHECKPOINT_FREE_BYTES = 20_000_000_000
MTP_PRECHECK_PROMPTS = 2
MTP_PRECHECK_NEW_TOKENS = 16
if MTP_UNFREEZE:
    MTP_TRAIN_STEPS = (50_000_000 + MTP_STEP_TOKENS - 1) // MTP_STEP_TOKENS
    MTP_LR = 1e-5
    MTP_EMBED_LR = 1e-4
    MTP_WARMUP_STEPS = 30
    MTP_WEIGHT_DECAY = 0.01
    MTP_EVAL_INTERVAL = (10_000_000 + MTP_STEP_TOKENS - 1) // MTP_STEP_TOKENS
    MTP_EVAL_PROMPTS = 64
    MAX_HOURS = 4
if SUITE in {'mtp', 'mtp-unfreeze', 'mtp-parallel'}:
    BLOCK_SIZE = MTP_CONTEXT
    GLOBAL_BATCH_SIZE = MTP_GLOBAL_BATCH
    TOKENS_PER_STEP = MTP_STEP_TOKENS
    TRAIN_TOKENS = MTP_TRAIN_STEPS * TOKENS_PER_STEP
    PROBE_STEPS = MTP_PROBE_STEPS
    experiment_configs = [ExperimentConfig(f's_sps_mtp_n{n}_emb_1b', 's_sps_w0_20b', horizons=n)
                          for n in (2, 3, 4)]
    if MTP_UNFREEZE:
        experiment_configs = [ExperimentConfig(f's_sps_mtp_n{n}_unfreeze_ce_50m', 's_sps_w0_20b', horizons=n)
                              for n in (2, 3, 4)]


def collect_experiments(filter_name=None):
    return [e for e in experiment_configs if filter_name is None or filter_name in e.name]


def overrides(exp, *, probe=False):
    root = SANITY_ROOT if probe else RUN_ROOT
    return [
        f'+experiment={exp.recipe}', f'system.data_root={root}',
        f'experiment.name={exp.name}', f'logging.wandb_run_name={exp.name}',
        'logging.wandb_log=false', 'system.compile=false',
        'training.micro_batch_size=6', f'training.global_batch_size={GLOBAL_BATCH_SIZE}',
        f'training.chunk_size_units={CHUNK_SIZE_UNITS}',
        f'training.max_tokens={PROBE_STEPS * TOKENS_PER_STEP if probe else TRAIN_TOKENS}',
        f'training.eval_total_tokens={TOKENS_PER_STEP if probe else 1048576}',
        f'training.eval_interval_tokens={TRAIN_TOKENS if probe else (250000000 if SUITE == "10b" else 50000000)}',
        f'training.skip_first_eval={str(probe).lower()}',
        f'training.init_from={"scratch" if probe else "resume"}',
        f'training.save_every={TRAIN_TOKENS if probe else (1000000000 if SUITE == "10b" else 250000000)}',
        f'training.rolling_save_every={TRAIN_TOKENS if probe else (250000000 if SUITE == "10b" else 100000000)}',
        f'scheduler.warmup_tokens={TOKENS_PER_STEP if probe else (100000000 if SUITE == "10b" else 10000000)}',
        f'scheduler.lr_decay_tokens={TOKENS_PER_STEP if probe else (1000000000 if SUITE == "10b" else 100000000)}',
    ]


def train_command(exp, *, probe=False):
    if SUITE == 'spec-baselines':
        if not exp.trainable: raise ValueError('This baseline does not require training')
        return [str(PYTHON), '-m', 'torch.distributed.run', '--standalone',
                f'--nproc_per_node={exp.num_gpus}', '-m', 'scripts.spec_baseline_train',
                '--filter', exp.name, *(['--probe'] if probe else [])]
    if SUITE in {'mtp', 'mtp-unfreeze', 'mtp-parallel'}:
        return [str(PYTHON), '-m', 'torch.distributed.run', '--standalone',
                f'--nproc_per_node={exp.num_gpus}', '-m', 'scripts.mtp_train',
                '--filter', exp.name, *(['--probe'] if probe else [])]
    return [str(PYTHON), '-m', 'torch.distributed.run', '--standalone',
            f'--nproc_per_node={exp.num_gpus}', 'scripts/train.py', *overrides(exp, probe=probe)]


def eval_command(exp, task):
    return [str(PYTHON), 'scripts/evaluate.py', *overrides(exp), '+checkpoint=final',
            f'+tasks={task}', '+eval_batch_size=6', '+num_fewshot=0']


def output_dir(exp):
    return RUN_ROOT / 'out' / exp.name


def design_signature():
    """Reject stale completion/probe markers after a design change."""
    repo = Path(__file__).resolve().parent
    inputs = sorted([*repo.glob('conf/**/*.yaml'), *repo.glob('src/**/*.py'),
                     repo / 'scripts/train.py', repo / 'scripts/evaluate.py'])
    if SUITE in {'mtp', 'mtp-unfreeze', 'mtp-parallel', 'spec-baselines', 'mtp-unified'}:
        inputs += [repo / 'experiments.py', *repo.glob('scripts/mtp*.py'), *repo.glob('scripts/spec_baseline*.py')]
    sources = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in inputs}
    design = dict(repo=DATASET_REPO, revision=DATASET_REVISION, files=DATA_FILES,
                  val_fraction=VAL_FRACTION, val_seed=VAL_SEED,
                  commands=[train_command(e) for e in experiment_configs if e.trainable],
                  probes=[train_command(e, probe=True) for e in experiment_configs if e.trainable],
                  tasks=TASKS, sources=sources)
    return hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()


def training_marker(exp):
    """Evaluation-only suites consume immutable completed runs."""
    if SUITE in {'mtp-parallel','mtp-unified'} or (SUITE == 'spec-baselines' and exp.recipe == 'existing_mtp'):
        return COMMON_ROOT / 'mtp-unfreeze/runs/out' / (exp.reference_name or exp.name) / 'trained.json'
    if not exp.trainable:
        return None
    return output_dir(exp) / 'trained.json'


# Parallel verifier numerical audit uses held-out prompts outside timed regions.
MTP_PARALLEL_PROB_ATOL = 0.005
MTP_PARALLEL_AUDIT_PROMPTS = 8

MTP_PARALLEL_TIMEOUT_SECONDS = 4 * 3600


# Additional proposal baselines. Training budgets count input text tokens once.
BASELINE_CONTEXT = 1024
BASELINE_GLOBAL_BATCH = 64
BASELINE_MICRO_BATCH = 2
BASELINE_STEP_TOKENS = BASELINE_CONTEXT * BASELINE_GLOBAL_BATCH
BASELINE_STEPS = 15259 + 763
BASELINE_PROBE_STEPS = 16
BASELINE_RESERVE_HORIZONS = 5
BASELINE_MEDUSA_HEADS = 4
BASELINE_LR = {'medusa': 1e-3, 'ar_tiny': 3e-4}
BASELINE_WARMUP = 100
BASELINE_WEIGHT_DECAY = .01
BASELINE_BETAS = (.9, .95)
BASELINE_CLIP = 1.
BASELINE_EVAL_INTERVAL = 500
BASELINE_VAL_TOKENS = 131072
BASELINE_MAX_HOURS = 24
BASELINE_AR_ARGS = dict(n_layer=4,n_head=4,hidden_size=256,intermediate_size=768)
BASELINE_LOOKUP_MATCH = 8
BASELINE_EVAL_HORIZONS = (2,3,4)
BASELINE_EVAL_PROMPTS = 64
BASELINE_PROMPT_TOKENS = 128
BASELINE_NEW_TOKENS = 128
BASELINE_WARMUP_PASSES = 2
BASELINE_EVAL_TIMEOUT = 4*3600
if SUITE == 'spec-baselines':
    MAX_HOURS = BASELINE_MAX_HOURS
    TRAIN_TOKENS = BASELINE_STEPS*BASELINE_STEP_TOKENS
    experiment_configs = [
        ExperimentConfig('medusa_heads_ce', 'medusa', horizons=4),
        ExperimentConfig('tiny_ar_ce', 'ar_tiny', horizons=4),
        ExperimentConfig('history_lookup', 'ngram', horizons=4,trainable=False),
        *[ExperimentConfig(f'mtp_n{n}_control','existing_mtp',horizons=n,
            reference_name=f's_sps_mtp_n{n}_unfreeze_ce_50m',trainable=False) for n in (2,3,4)],
    ]

# Unified self-speculation consumes full continuations; no additional training.
UNIFIED_PROMPTS = 64
UNIFIED_PROMPT_TOKENS = 128
UNIFIED_NEW_TOKENS = 128
UNIFIED_WARMUP_PASSES = 2
UNIFIED_TIMEOUT = 4*3600
if SUITE == 'mtp-unified':
    experiment_configs = [ExperimentConfig(f'mtp_n{n}_self','existing_mtp',horizons=n,
        reference_name=f's_sps_mtp_n{n}_unfreeze_ce_50m',trainable=False) for n in (2,3,4)]
