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
    DATASET_REPO,
    DATASET_REVISION,
    EXISTING_HF_CACHES,
    MAX_HOURS,
    TASKS,
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

DATA_FILES = pilot.data_files(SUITE)
MIN_FREE_BYTES = pilot.min_free_bytes(SUITE)
RUNTIME_SAFETY_FACTOR = pilot.runtime_safety_factor(SUITE)

experiment_configs = (
    pilot.zero_window_longer_training() if SUITE == "10b" else pilot.zero_window_pilot()
)
_training = experiment_configs[0].training
GLOBAL_BATCH_SIZE = _training.global_batch_size
BLOCK_SIZE = _training.block_size
TOKENS_PER_STEP = GLOBAL_BATCH_SIZE * BLOCK_SIZE
CHUNK_SIZE_UNITS = _training.chunk_size_units
TRAINING_SEED = _training.seed
TRAIN_TOKENS = _training.max_tokens
PROBE_STEPS = _training.probe_steps
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
    config = exp.config(root, probe=probe)
    args = [f"+experiment={exp.recipe}"]
    for group in ("experiment", "training", "scheduler", "system", "logging"):
        for key, value in config[group].items():
            if key != "tags":
                literal = (
                    json.dumps(value)
                    if isinstance(value, bool) or value is None
                    else str(value)
                )
                args.append(f"{group}.{key}={literal}")
    return args


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
        "-m",
        "scripts.pilot.named_worker",
        "train",
        exp.name,
        *(["--probe"] if probe else []),
    ]


def eval_command(exp, task):
    return [
        str(PYTHON),
        "-m",
        "scripts.pilot.named_worker",
        "eval",
        exp.name,
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
            definitions / "mtp_design.py",
            definitions / "mtp_unified.py",
            definitions / "spec_baselines.py",
            *repo.glob("scripts/speculative/mtp*.py"),
            *repo.glob("scripts/speculative/spec_baseline*.py"),
        ]
    if SUITE in AR_SPS_SUITES:
        inputs += [
            definitions / "ar_sps_config.py",
            definitions / "ar_sps.py",
            *repo.glob("scripts/adaptation/ar_sps*.py"),
        ]
    if SUITE in MMLU_SUITES:
        inputs += [
            definitions / "mmlu_config.py",
            definitions / "mmlu.py",
            *repo.glob("scripts/evaluation/mmlu*.py"),
            repo / "scripts/report_mmlu_stem.py",
        ]
    if SUITE == INSTRUCT_SUITE:
        inputs += [
            definitions / "instruct_config.py",
            definitions / "mmlu_config.py",
            definitions / "mmlu.py",
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
    if SUITE in {"1b", "10b"}:
        design["evaluation_commands"] = [
            eval_command(exp, task) for exp in experiment_configs for task in TASKS
        ]
        design["configs"] = [
            (exp.config(RUN_ROOT), exp.config(SANITY_ROOT, probe=True))
            for exp in experiment_configs
        ]
        worker_path = repo / "scripts/pilot/named_worker.py"
        if worker_path.exists():
            design["pilot_worker"] = hashlib.sha256(
                worker_path.read_bytes()
            ).hexdigest()
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
