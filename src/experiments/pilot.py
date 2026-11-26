"""FineWeb-Edu 1B and 10B pilot definitions."""

from pathlib import Path

from .suite import SUITE
from .types import ExperimentConfig

DATASET_REPO = "HuggingFaceFW/fineweb-edu"
DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
# Three complete sample-10BT shards supply roughly two billion source tokens.
DATA_FILES = tuple(f"sample/10BT/{i:03d}_00000.parquet" for i in range(3))
if SUITE == "10b":
    # Spread the larger corpus across crawl partitions, with >10B GPT-2 tokens
    # after the held-out split. Reuse the same pinned repository/cache.
    DATA_FILES = tuple(f"sample/100BT/{i:03d}_00000.parquet" for i in range(14)) + (
        "sample/100BT/000_00001.parquet",
        "sample/100BT/001_00001.parquet",
    )
EXISTING_HF_CACHES = (
    Path("/home/jovyan/.cache/huggingface/hub"),
    Path("/mnt/virtual_ai0001053-00054_SR004-nfs2/.cache/huggingface/hub"),
    Path("/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/hf-cache/hub"),
    Path("/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/HF_CACHE_NFS1/hub"),
)
TASKS = ("arc_easy", "hellaswag", "piqa", "sciq", "lambada_openai")
GLOBAL_BATCH_SIZE = 96
BLOCK_SIZE = 4096
TOKENS_PER_STEP = GLOBAL_BATCH_SIZE * BLOCK_SIZE
CHUNK_SIZE_UNITS = 1024
TRAINING_SEED = 1337
TRAIN_TOKENS = 2544 * TOKENS_PER_STEP  # 1,000,341,504; exact update boundary
if SUITE == "10b":
    TRAIN_TOKENS *= 10
PROBE_STEPS = 16
MAX_HOURS = 24
VAL_FRACTION = 0.01
VAL_SEED = 2357
MIN_FREE_BYTES = (120 if SUITE == "10b" else 40) * 10**9
# The long run has a complete prior run for calibration rather than only a
# short probe. Keep a margin plus one hour for startup and checkpoint I/O.
RUNTIME_SAFETY_FACTOR = 1.15 if SUITE == "10b" else 1.5


experiment_configs = [
    ExperimentConfig("s_standard_pilot_1b", "s_full_attention_20b"),
    ExperimentConfig("s_sps_w0_pilot_1b", "s_sps_w0_20b"),
]
if SUITE == "10b":
    experiment_configs = [
        ExperimentConfig(
            "s_standard_10b",
            "s_full_attention_20b",
            reference_name="s_standard_pilot_1b",
        ),
        ExperimentConfig(
            "s_sps_w0_10b", "s_sps_w0_20b", reference_name="s_sps_w0_pilot_1b"
        ),
    ]
