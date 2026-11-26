"""Compatibility interface for suite-scoped MLSpace worker processes."""
from pathlib import Path
import inspect
import hashlib
import json
import os

from experiment_defs.paths import COMMON_ROOT
from experiment_defs import mtp, speculative_baselines, unified
from experiment_defs.speculative_run import ExperimentConfig
from experiment_defs.zero_window import (DATASET_REPO, DATASET_REVISION, data_files,
                                         zero_window_pilot, zero_window_longer_training)
from scripts.design_compatibility import canonical_signature
from scripts.suite_registry import (AR_SPS_SUITES, INSTRUCT_SUITE, LARGE_AR_SPS_SUITES,
                                    MMLU_SUITES, WORKER_MODULES)

SUITE = os.environ.get('SPS_EXPERIMENT_SUITE', '1b')
if SUITE not in WORKER_MODULES:
    raise ValueError(f'Unknown SPS_EXPERIMENT_SUITE={SUITE!r}')
ARTIFACTS_ROOT = COMMON_ROOT if SUITE == '1b' else COMMON_ROOT / SUITE
RUN_ROOT = ARTIFACTS_ROOT / 'runs'
SANITY_ROOT = ARTIFACTS_ROOT / 'sanity'
STAGING_ROOT = COMMON_ROOT / 'code'
HF_HOME = COMMON_ROOT / 'hf'
PYTHON = Path('/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/sps/.venv/bin/python')
INSTANCE_TYPES_BY_NUM_GPUS = {1: 'a100.1gpu', 4: 'a100.4gpu'}
EVAL_NUM_GPUS = 1  # Project policy: evaluation workers always use one GPU.
BASE_IMAGE = 'cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41'
DATA_FILES = data_files(SUITE)
EXISTING_HF_CACHES = (
    Path('/home/jovyan/.cache/huggingface/hub'),
    Path('/mnt/virtual_ai0001053-00054_SR004-nfs2/.cache/huggingface/hub'),
    Path('/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/hf-cache/hub'),
    Path('/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/HF_CACHE_NFS1/hub'),
)
TASKS = ('arc_easy', 'hellaswag', 'piqa', 'sciq', 'lambada_openai')
experiment_configs = zero_window_longer_training() if SUITE == '10b' else zero_window_pilot()
_pilot_training = experiment_configs[0].training
GLOBAL_BATCH_SIZE = _pilot_training.global_batch_size
BLOCK_SIZE = _pilot_training.block_size
TOKENS_PER_STEP = GLOBAL_BATCH_SIZE * BLOCK_SIZE
CHUNK_SIZE_UNITS = _pilot_training.chunk_size_units
TRAINING_SEED = _pilot_training.seed
TRAIN_TOKENS = _pilot_training.max_tokens
PROBE_STEPS = _pilot_training.probe_steps
MAX_HOURS = 24
VAL_FRACTION = 0.01
VAL_SEED = 2357
MIN_FREE_BYTES = (120 if SUITE == '10b' else 40) * 10**9
# The long run has a complete prior run for calibration rather than only a
# short probe. Keep a margin plus one hour for startup and checkpoint I/O.
RUNTIME_SAFETY_FACTOR = 1.15 if SUITE == '10b' else 1.5


# Legacy worker modules still import these suite-scoped settings.
for family in (mtp, speculative_baselines, unified):
    globals().update(vars(family.configuration(SUITE)))


def collect_experiments(filter_name=None):
    return [e for e in experiment_configs if filter_name is None or filter_name in e.name]


def overrides(exp, *, probe=False):
    root = SANITY_ROOT if probe else RUN_ROOT
    config = exp.config(root, probe=probe)
    args = [f'+experiment={exp.recipe}']
    for group in ('experiment', 'training', 'scheduler', 'system', 'logging'):
        for key, value in config[group].items():
            if key != 'tags':
                literal = json.dumps(value) if isinstance(value, bool) or value is None else str(value)
                args.append(f'{group}.{key}={literal}')
    return args


def train_command(exp, *, probe=False):
    if SUITE == INSTRUCT_SUITE:
        if not exp.trainable:
            raise ValueError('The original instruct model does not require training')
        return [str(PYTHON), '-m', 'torch.distributed.run', '--standalone',
                f'--nproc_per_node={exp.num_gpus}', '-m', 'scripts.instruct_train',
                '--filter', exp.name, *(['--probe'] if probe else [])]
    if SUITE in AR_SPS_SUITES:
        return [str(PYTHON), '-m', 'torch.distributed.run', '--standalone',
                f'--nproc_per_node={exp.num_gpus}', '-m', 'scripts.ar_sps_train',
                '--filter', exp.name, *(['--probe'] if probe else [])]
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
            f'--nproc_per_node={exp.num_gpus}', '-m', 'scripts.pilot.named_worker',
            'train', exp.name, *(['--probe'] if probe else [])]


def eval_command(exp, task):
    return [str(PYTHON), '-m', 'scripts.pilot.named_worker', 'eval', exp.name, '+checkpoint=final',
            f'+tasks={task}', '+eval_batch_size=6', '+num_fewshot=0']


def output_dir(exp):
    return RUN_ROOT / 'out' / exp.name


def design_signature():
    """Reject stale completion/probe markers after a design change."""
    repo = Path(__file__).resolve().parent
    inputs = sorted([*repo.glob('conf/**/*.yaml'), *repo.glob('src/**/*.py'),
                     repo / 'scripts/train.py', repo / 'scripts/evaluate.py'])
    if SUITE in {'mtp', 'mtp-unfreeze', 'mtp-parallel', 'spec-baselines', 'mtp-unified'}:
        inputs += [repo / 'experiments.py', *repo.glob('scripts/speculative/mtp*.py'), *repo.glob('scripts/speculative/spec_baseline*.py')]
    if SUITE in AR_SPS_SUITES:
        inputs += [repo / 'experiments.py', *repo.glob('scripts/adaptation/ar_sps*.py')]
    if SUITE in MMLU_SUITES:
        inputs += [repo / 'experiments.py', *repo.glob('scripts/evaluation/mmlu*.py'), repo / 'scripts/report_mmlu_stem.py']
    if SUITE == INSTRUCT_SUITE:
        inputs += [repo / 'experiments.py', *repo.glob('scripts/instruct*.py'), *repo.glob('scripts/instruction/instruct*.py'),
                   *repo.glob('scripts/evaluation/mmlu*.py'), repo / 'scripts/report_mmlu_stem.py',
                   repo / 'scripts/mlspace_launch.py', repo / 'pyproject.toml', repo / 'uv.lock',
                   repo / 'requirements-instruct.txt']
    # Capture moved family definitions in the same identity as their workers.
    from experiment_defs.registry import CONFIGURATIONS
    if SUITE in CONFIGURATIONS:
        inputs.append(Path(inspect.getfile(CONFIGURATIONS[SUITE])))
    if SUITE in {'mtp', 'mtp-unfreeze', 'mtp-parallel', 'mtp-unified', 'spec-baselines'}:
        inputs.extend([repo / 'experiment_defs/mtp.py', repo / 'experiment_defs/speculative_run.py'])
    if SUITE in MMLU_SUITES or SUITE == INSTRUCT_SUITE:
        inputs.append(repo / 'experiment_defs/mmlu.py')
    # Keep the logical filenames used in historical design manifests.
    sources = {(f'scripts/{p.name}' if p.is_relative_to(repo / 'scripts') else str(p.relative_to(repo))): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in inputs}
    design = dict(suite=SUITE, worker=WORKER_MODULES[SUITE], repo=DATASET_REPO, revision=DATASET_REVISION, files=DATA_FILES,
                  val_fraction=VAL_FRACTION, val_seed=VAL_SEED,
                  commands=[train_command(e) for e in experiment_configs if e.trainable],
                  probes=[train_command(e, probe=True) for e in experiment_configs if e.trainable],
                  tasks=TASKS, sources=sources)
    if SUITE in {'1b', '10b'}:
        design['configs'] = [(exp.config(RUN_ROOT), exp.config(SANITY_ROOT, probe=True))
                             for exp in experiment_configs]
        worker_path = repo / 'scripts/pilot/named_worker.py'
        if worker_path.exists():
            design['pilot_worker'] = hashlib.sha256(worker_path.read_bytes()).hexdigest()
    if SUITE in LARGE_AR_SPS_SUITES:
        from scripts import ar_sps_config as c
        design['large_ce'] = dict(suite=SUITE, model=c.MODEL, revision=c.REVISION,
                                  micro_batch=c.MICRO_BATCH, eval_batch=c.EVAL_BATCH)
    if SUITE == INSTRUCT_SUITE:
        from scripts import instruct_config as c
        design['instruct'] = dict(suite=SUITE, model=c.MODEL, revision=c.REVISION)
    digest = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    return canonical_signature(digest)


def training_marker(exp):
    """Evaluation-only suites consume immutable completed runs."""
    if SUITE in {'mtp-parallel','mtp-unified'} or (SUITE == 'spec-baselines' and exp.recipe == 'existing_mtp'):
        return COMMON_ROOT / 'mtp-unfreeze/runs/out' / (exp.reference_name or exp.name) / 'trained.json'
    if SUITE in MMLU_SUITES and exp.variant != 'base':
        return COMMON_ROOT / exp.suite / 'runs/out' / exp.arm / 'trained.json'
    if not exp.trainable:
        return None
    return output_dir(exp) / 'trained.json'


if SUITE in AR_SPS_SUITES:
    from scripts.ar_sps_config import experiment_configs

if SUITE in MMLU_SUITES:
    from scripts.mmlu_config import experiment_configs
    TASKS = (SUITE.replace('-', '_'),)

if SUITE == INSTRUCT_SUITE:
    from scripts.instruct_config import experiment_configs
