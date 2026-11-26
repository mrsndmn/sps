"""Select experiment definitions and build commands for the shared MLSpace launchers."""

import hashlib
import json
from pathlib import Path

from orchestration.design_compatibility import canonical_signature
from orchestration.suite_registry import (
    AR_SPS_SUITES,
    INSTRUCT_SUITE,
    LARGE_AR_SPS_SUITES,
    MMLU_SUITES,
    WORKER_SCRIPTS,
)
from experiments import mtp, mtp_unified, pilot, spec_baselines
from experiments.environment import (
    BASE_IMAGE,
    COMMON_ROOT,
    EVAL_NUM_GPUS,
    HF_HOME,
    INSTANCE_TYPES_BY_NUM_GPUS,
    PYTHON,
    STAGING_ROOT,
)
from experiments.pilot import (
    DATASET_REPO,
    DATASET_REVISION,
    EXISTING_HF_CACHES,
    MAX_HOURS,
    TASKS,
    VAL_FRACTION,
    VAL_SEED,
)
from experiments.suite import SUITE
from experiments.types import ExperimentConfig

REPO_ROOT = Path(__file__).resolve().parents[2]

if SUITE not in WORKER_SCRIPTS:
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
    from experiments.ar_sps_config import experiment_configs
elif SUITE in MMLU_SUITES:
    from experiments.mmlu_config import experiment_configs

    TASKS = (SUITE.replace("-", "_"),)
elif SUITE == INSTRUCT_SUITE:
    from experiments.instruct_config import experiment_configs


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


def train_command(exp, *, probe=False, data_root=None, dest=None):
    if SUITE == INSTRUCT_SUITE:
        if not exp.trainable:
            raise ValueError("The original instruct model does not require training")
        return [
            str(PYTHON),
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={exp.num_gpus}",
            "scripts/instruction/instruct_train.py",
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
            "scripts/adaptation/ar_sps_train.py",
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
            "scripts/speculative/spec_baseline_train.py",
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
            "scripts/speculative/mtp_train.py",
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
        *pilot_cli(exp, data_root or (SANITY_ROOT if probe else RUN_ROOT), probe=probe),
        "--out_dir", str(dest or ((data_root or SANITY_ROOT) / "out" / exp.name if probe else exp.output_dir)),
    ]


def pilot_cli(exp, root, *, probe=False):
    from experiments.worker_args import cli_args
    return cli_args(exp.arguments(root, probe=probe))


def eval_command(exp, task, *, result=None, checkpoint=None):
    return [
        str(PYTHON), "scripts/evaluate.py", *pilot_cli(exp, RUN_ROOT),
        "--out_dir", str(exp.output_dir),
        *(["--checkpoint_path", str(checkpoint)] if checkpoint else ["--checkpoint", "final"]),
        "--tasks", task,
        "--eval_batch_size", "6", "--num_fewshot", "0",
        *(["--output_path", str(result)] if result else []),
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
            repo / "src/training/train.py",
            repo / "src/modeling/evaluation/evaluate.py",
        ]
    )
    inputs += [
        repo / "src/orchestration/experiment_suite.py",
        repo / "src/orchestration/suite_registry.py",
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
            *repo.glob("src/workloads/speculative/mtp*.py"),
            *repo.glob("src/workloads/speculative/spec_baseline*.py"),
        ]
    if SUITE in AR_SPS_SUITES:
        inputs += [
            definitions / "ar_sps_config.py",
            definitions / "ar_sps.py",
            *repo.glob("src/workloads/adaptation/ar_sps*.py"),
        ]
    if SUITE in MMLU_SUITES:
        inputs += [
            definitions / "mmlu_config.py",
            definitions / "mmlu.py",
            *repo.glob("src/workloads/evaluation/mmlu*.py"),
            repo / "src/workloads/evaluation/report_mmlu_stem.py",
        ]
    if SUITE == INSTRUCT_SUITE:
        inputs += [
            definitions / "instruct_config.py",
            repo / "src/data/instruct_data.py",
            definitions / "mmlu_config.py",
            definitions / "mmlu.py",
            *repo.glob("src/workloads/instruction/instruct*.py"),
            *repo.glob("src/workloads/evaluation/mmlu*.py"),
            repo / "src/workloads/evaluation/report_mmlu_stem.py",
            repo / "src/orchestration/mlspace_launch.py",
            repo / "pyproject.toml",
            repo / "uv.lock",
            repo / "requirements-instruct.txt",
        ]
    scripts = repo / "scripts"
    worker_script = Path(WORKER_SCRIPTS[SUITE])
    inputs += [
        scripts / "train.py",
        scripts / "evaluate.py",
        repo / worker_script,
        repo / "src/workloads" / worker_script.relative_to("scripts"),
    ]
    sources = {
        str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in inputs
    }
    design = dict(
        suite=SUITE,
        worker=WORKER_SCRIPTS[SUITE],
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
    if SUITE in LARGE_AR_SPS_SUITES:
        from experiments import ar_sps_config as c

        design["large_ce"] = dict(
            suite=SUITE,
            model=c.MODEL,
            revision=c.REVISION,
            micro_batch=c.MICRO_BATCH,
            eval_batch=c.EVAL_BATCH,
        )
    if SUITE == INSTRUCT_SUITE:
        from experiments import instruct_config as c

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
