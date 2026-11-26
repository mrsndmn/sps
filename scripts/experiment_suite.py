"""Select experiment definitions and build commands for the shared MLSpace launchers."""

import hashlib
import json
from pathlib import Path

from scripts.design_compatibility import canonical_signature
from scripts.suite_registry import (
    AR_SPS_SUITES,
    INSTRUCT_SUITE,
    LARGE_AR_SPS_SUITES,
    MMLU_SUITES,
    WORKER_MODULES,
)
from src.experiments import mtp, mtp_unified, pilot, spec_baselines
from src.experiments.environment import (
    BASE_IMAGE,
    COMMON_ROOT,
    EVAL_NUM_GPUS,
    HF_HOME,
    INSTANCE_TYPES_BY_NUM_GPUS,
    PYTHON,
    STAGING_ROOT,
)
from src.experiments.pilot import (
    BLOCK_SIZE,
    CHUNK_SIZE_UNITS,
    DATA_FILES,
    DATASET_REPO,
    DATASET_REVISION,
    EXISTING_HF_CACHES,
    GLOBAL_BATCH_SIZE,
    MAX_HOURS,
    MIN_FREE_BYTES,
    PROBE_STEPS,
    RUNTIME_SAFETY_FACTOR,
    TASKS,
    TOKENS_PER_STEP,
    TRAIN_TOKENS,
    TRAINING_SEED,
    VAL_FRACTION,
    VAL_SEED,
)
from src.experiments.suite import SUITE
from src.experiments.types import ExperimentConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

if SUITE not in WORKER_MODULES:
    raise ValueError(f"Unknown SPS_EXPERIMENT_SUITE={SUITE!r}")

ARTIFACTS_ROOT = COMMON_ROOT if SUITE == "1b" else COMMON_ROOT / SUITE
RUN_ROOT = ARTIFACTS_ROOT / "runs"
SANITY_ROOT = ARTIFACTS_ROOT / "sanity"

experiment_configs = pilot.experiment_configs
if SUITE in {"mtp", "mtp-unfreeze", "mtp-parallel"}:
    experiment_configs = mtp.experiment_configs
    BLOCK_SIZE = mtp.MTP_CONTEXT
    GLOBAL_BATCH_SIZE = mtp.MTP_GLOBAL_BATCH
    TOKENS_PER_STEP = mtp.MTP_STEP_TOKENS
    TRAIN_TOKENS = mtp.MTP_TRAIN_STEPS * TOKENS_PER_STEP
    PROBE_STEPS = mtp.MTP_PROBE_STEPS
    if mtp.MTP_UNFREEZE:
        MAX_HOURS = mtp.MTP_UNFREEZE_MAX_HOURS
elif SUITE == "spec-baselines":
    experiment_configs = spec_baselines.experiment_configs
    MAX_HOURS = spec_baselines.BASELINE_MAX_HOURS
    TRAIN_TOKENS = spec_baselines.BASELINE_STEPS * spec_baselines.BASELINE_STEP_TOKENS
elif SUITE == "mtp-unified":
    experiment_configs = mtp_unified.experiment_configs
elif SUITE in AR_SPS_SUITES:
    from src.experiments.ar_sps_config import experiment_configs
elif SUITE in MMLU_SUITES:
    from src.experiments.mmlu_config import experiment_configs

    TASKS = (SUITE.replace("-", "_"),)
elif SUITE == INSTRUCT_SUITE:
    from src.experiments.instruct_config import experiment_configs


def collect_experiments(filter_name=None):
    return [
        e for e in experiment_configs if filter_name is None or filter_name in e.name
    ]


def overrides(exp, *, probe=False):
    root = SANITY_ROOT if probe else RUN_ROOT
    return [
        f"+experiment={exp.recipe}",
        f"system.data_root={root}",
        f"experiment.name={exp.name}",
        f"logging.wandb_run_name={exp.name}",
        "logging.wandb_log=false",
        "system.compile=false",
        "training.micro_batch_size=6",
        f"training.global_batch_size={GLOBAL_BATCH_SIZE}",
        f"training.chunk_size_units={CHUNK_SIZE_UNITS}",
        f"training.max_tokens={PROBE_STEPS * TOKENS_PER_STEP if probe else TRAIN_TOKENS}",
        f"training.eval_total_tokens={TOKENS_PER_STEP if probe else 1048576}",
        f"training.eval_interval_tokens={TRAIN_TOKENS if probe else (250000000 if SUITE == '10b' else 50000000)}",
        f"training.skip_first_eval={str(probe).lower()}",
        f"training.init_from={'scratch' if probe else 'resume'}",
        f"training.save_every={TRAIN_TOKENS if probe else (1000000000 if SUITE == '10b' else 250000000)}",
        f"training.rolling_save_every={TRAIN_TOKENS if probe else (250000000 if SUITE == '10b' else 100000000)}",
        f"scheduler.warmup_tokens={TOKENS_PER_STEP if probe else (100000000 if SUITE == '10b' else 10000000)}",
        f"scheduler.lr_decay_tokens={TOKENS_PER_STEP if probe else (1000000000 if SUITE == '10b' else 100000000)}",
    ]


def train_command(exp, *, probe=False):
    if SUITE == INSTRUCT_SUITE:
        if not exp.trainable:
            raise ValueError("The original instruct model does not require training")
        return [
            str(PYTHON),
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={exp.num_gpus}",
            "-m",
            "scripts.instruct_train",
            "--filter",
            exp.name,
            *(["--probe"] if probe else []),
        ]
    if SUITE in AR_SPS_SUITES:
        return [
            str(PYTHON),
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={exp.num_gpus}",
            "-m",
            "scripts.ar_sps_train",
            "--filter",
            exp.name,
            *(["--probe"] if probe else []),
        ]
    if SUITE == "spec-baselines":
        if not exp.trainable:
            raise ValueError("This baseline does not require training")
        return [
            str(PYTHON),
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={exp.num_gpus}",
            "-m",
            "scripts.spec_baseline_train",
            "--filter",
            exp.name,
            *(["--probe"] if probe else []),
        ]
    if SUITE in {"mtp", "mtp-unfreeze", "mtp-parallel"}:
        return [
            str(PYTHON),
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={exp.num_gpus}",
            "-m",
            "scripts.mtp_train",
            "--filter",
            exp.name,
            *(["--probe"] if probe else []),
        ]
    return [
        str(PYTHON),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={exp.num_gpus}",
        "scripts/train.py",
        *overrides(exp, probe=probe),
    ]


def eval_command(exp, task):
    return [
        str(PYTHON),
        "scripts/evaluate.py",
        *overrides(exp),
        "+checkpoint=final",
        f"+tasks={task}",
        "+eval_batch_size=6",
        "+num_fewshot=0",
    ]


def output_dir(exp):
    return RUN_ROOT / "out" / exp.name


def design_signature():
    """Reject stale completion/probe markers after a design change."""
    repo = REPO_ROOT
    definitions = repo / "src/experiments"
    inputs = sorted(
        [
            *repo.glob("conf/**/*.yaml"),
            *repo.glob("src/modeling/**/*.py"),
            *repo.glob("src/training/**/*.py"),
            *repo.glob("src/data/**/*.py"),
            *repo.glob("src/plotting/**/*.py"),
            repo / "scripts/train.py",
            repo / "scripts/evaluate.py",
        ]
    )
    inputs += [
        repo / "scripts/experiment_suite.py",
        repo / "scripts/suite_registry.py",
        definitions / "environment.py",
        definitions / "pilot.py",
        definitions / "types.py",
        definitions / "suite.py",
    ]
    if SUITE in {
        "mtp",
        "mtp-unfreeze",
        "mtp-parallel",
        "spec-baselines",
        "mtp-unified",
    }:
        inputs += [
            definitions / "mtp.py",
            definitions / "mtp_unified.py",
            definitions / "spec_baselines.py",
            *repo.glob("scripts/speculative/mtp*.py"),
            *repo.glob("scripts/speculative/spec_baseline*.py"),
        ]
    if SUITE in AR_SPS_SUITES:
        inputs += [
            definitions / "ar_sps_config.py",
            *repo.glob("scripts/adaptation/ar_sps*.py"),
        ]
    if SUITE in MMLU_SUITES:
        inputs += [
            definitions / "mmlu_config.py",
            *repo.glob("scripts/evaluation/mmlu*.py"),
            repo / "scripts/report_mmlu_stem.py",
        ]
    if SUITE == INSTRUCT_SUITE:
        inputs += [
            definitions / "instruct_config.py",
            definitions / "mmlu_config.py",
            *repo.glob("scripts/instruct*.py"),
            *repo.glob("scripts/instruction/instruct*.py"),
            *repo.glob("scripts/evaluation/mmlu*.py"),
            repo / "scripts/report_mmlu_stem.py",
            repo / "scripts/mlspace_launch.py",
            repo / "pyproject.toml",
            repo / "uv.lock",
            repo / "requirements-instruct.txt",
        ]
    # Keep the logical filenames used in historical design manifests.
    sources = {
        (
            f"scripts/{p.name}"
            if p.is_relative_to(repo / "scripts")
            else str(p.relative_to(repo))
        ): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in inputs
    }
    design = dict(
        suite=SUITE,
        worker=WORKER_MODULES[SUITE],
        repo=DATASET_REPO,
        revision=DATASET_REVISION,
        files=DATA_FILES,
        val_fraction=VAL_FRACTION,
        val_seed=VAL_SEED,
        commands=[train_command(e) for e in experiment_configs if e.trainable],
        probes=[
            train_command(e, probe=True) for e in experiment_configs if e.trainable
        ],
        tasks=TASKS,
        sources=sources,
    )
    if SUITE in LARGE_AR_SPS_SUITES:
        from scripts import ar_sps_config as c

        design["large_ce"] = dict(
            suite=SUITE,
            model=c.MODEL,
            revision=c.REVISION,
            micro_batch=c.MICRO_BATCH,
            eval_batch=c.EVAL_BATCH,
        )
    if SUITE == INSTRUCT_SUITE:
        from scripts import instruct_config as c

        design["instruct"] = dict(suite=SUITE, model=c.MODEL, revision=c.REVISION)
    digest = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    return canonical_signature(digest)


def training_marker(exp):
    """Evaluation-only suites consume immutable completed runs."""
    if SUITE in {"mtp-parallel", "mtp-unified"} or (
        SUITE == "spec-baselines" and exp.recipe == "existing_mtp"
    ):
        return (
            COMMON_ROOT
            / "mtp-unfreeze/runs/out"
            / (exp.reference_name or exp.name)
            / "trained.json"
        )
    if SUITE in MMLU_SUITES and exp.variant != "base":
        return COMMON_ROOT / exp.suite / "runs/out" / exp.arm / "trained.json"
    if not exp.trainable:
        return None
    return output_dir(exp) / "trained.json"
