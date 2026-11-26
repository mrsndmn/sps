"""Executable definitions for dflash repro train."""

from dataclasses import dataclass
from .environment import COMMON_ROOT
from .qwen4b_drafter_train import SEED as SEED

ROOT = COMMON_ROOT / "dflash-repro"

PROPOSALS = 15

WIDTH = PROPOSALS + 1

MAX_ANCHORS = 512

HEAD_CHUNK = 256

QUERY_BLOCK = 128

LR = 6e-4

WARMUP = 0.04

GAMMA = 7.0

PROBE_WARMUP = 6

PROBE_STEPS = 24

MILESTONES = (5000, 10000)


@dataclass(frozen=True)
class Experiment:
    name: str
    epochs: int
    global_batch: int
    num_gpus: int


experiment_configs = [Experiment("qwen3-4b-dflash-b16-scratch", 3, 24, 4)]

(EXPERIMENT,) = experiment_configs
