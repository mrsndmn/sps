"""Shared corpus and named MLSpace launches for zero-window pretraining."""
from experiments.environment import COMMON_ROOT
from experiments.launch import Launch
from experiments.paper import ZeroWindowReproduction
from experiments.pilot import data_files

FAMILY = "paper-sps"
ROOT = COMMON_ROOT / "paper-sps-scale-reproduction"
CORPUS = ROOT / "data" / "fineweb-edu"
PREFIX = COMMON_ROOT / "10b/runs/data/fineweb-edu"
PREFIX_FILES = data_files("10b")
FILES = PREFIX_FILES + tuple(f"sample/100BT/{i:03d}_00001.parquet" for i in range(2, 14)) + tuple(
    f"sample/100BT/{i:03d}_00002.parquet" for i in range(8)
)
PREPARE_FREE_BYTES = 120_000_000_000
SCRIPT = "scripts/pilot/paper_reproduction.py"
experiment_configs = ZeroWindowReproduction.experiment_configs()


def launches(family):
    yield Launch("data", "prepare", SCRIPT, ("data",))
    for run in experiment_configs:
        yield Launch(run.scale, "train", SCRIPT, (run.scale,))
