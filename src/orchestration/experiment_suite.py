"""Select experiment definitions and build commands for the shared MLSpace launchers."""

import os
import hashlib
import json
from pathlib import Path

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

if os.environ.get("SPS_WORK_PHASE") == "prepare":
    SANITY_ROOT = Path(os.environ["SPS_PHASE_WORK"]) / "sanity"
else:
    from experiments.phase_outputs import families as phase_families
    if SUITE in phase_families:
        prepared_sanity = phase_families[SUITE].path(ARTIFACTS_ROOT, "prepare") / "sanity"
        if prepared_sanity.is_dir():
            SANITY_ROOT = prepared_sanity

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
    BLOCK_SIZE = mtp.family_config.MTP_CONTEXT
    GLOBAL_BATCH_SIZE = mtp.family_config.MTP_GLOBAL_BATCH
    TOKENS_PER_STEP = mtp.family_config.MTP_STEP_TOKENS
    TRAIN_TOKENS = mtp.family_config.MTP_TRAIN_STEPS * TOKENS_PER_STEP
    PROBE_STEPS = mtp.family_config.MTP_PROBE_STEPS
    if mtp.family_config.MTP_UNFREEZE:
        MAX_HOURS = mtp.family_config.MTP_UNFREEZE_MAX_HOURS
elif SUITE == "spec-baselines":
    experiment_configs = spec_baselines.experiment_configs
    MAX_HOURS = spec_baselines.family_config.BASELINE_MAX_HOURS
    TRAIN_TOKENS = spec_baselines.family_config.BASELINE_STEPS * spec_baselines.family_config.BASELINE_STEP_TOKENS
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


def phase_work(phase, fallback):
    if os.environ.get("SPS_WORK_PHASE") == phase:
        return Path(os.environ["SPS_PHASE_WORK"])
    return fallback


def prepared_path():
    from experiments.phase_outputs import families
    current = phase_work("prepare", None)
    if current is not None:
        return current / "prepared.json"
    if SUITE in families:
        configured = families[SUITE].path(ARTIFACTS_ROOT, "prepare")
        if configured.exists() or configured != ARTIFACTS_ROOT / "prepared":
            return configured / "prepared.json"
    return ARTIFACTS_ROOT / "prepared.json"


def output_dir(exp):
    from experiments.phase_outputs import families
    final = (families[SUITE].path(ARTIFACTS_ROOT, "train", exp.name)
             if SUITE in families else RUN_ROOT / "out" / exp.name)
    return phase_work("train", final) if os.environ.get("SPS_PHASE_RUN") == exp.name else final


def published(path):
    work = os.environ.get("SPS_PHASE_WORK")
    if work and Path(path).is_relative_to(work):
        return Path(os.environ["SPS_PUBLISHED_OUTPUT"]) / Path(path).relative_to(work)
    return path


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
    if SUITE in {"1b", "10b"}:
        return exp.output_dir / "trained.json"
    return output_dir(exp) / "trained.json"
