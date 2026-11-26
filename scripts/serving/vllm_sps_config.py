"""Executable design for shared-backbone Qwen3 SPS-MTP and vLLM comparison."""

from dataclasses import dataclass
from pathlib import Path

from scripts import instruct_config as source

ROOT = Path(
    "/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/sps-artifacts/vllm-sps-qwen3"
)
SOURCE_ROOT = ROOT.parent / "ar-sps-qwen3-1.7b-instruct"
SOURCE_SIGNATURE = "761505551994ceec0a907401b3ee875ffea744b69bc94c7007b23c58181b466b"
PARENT = (
    ROOT.parent / "qwen3-instruct-cosine/runs/out/qwen3_instruct_staged_sps_ce/final.pt"
)
PARENT_SIGNATURE = "7408b3ccf0455895e306d0d5272e2fad41d3e70ae8c6fbb5612210d6d18347ae"
PARENT_STEP = 512
VLLM_WORKTREE = ROOT.parent.parent / "vllm-sps"
VLLM_BASE = "3fe0d44219471ee5b178036a953b6f0c0c2bf5f6"
VLLM_COMMIT = "65ff372c9f6fbdeebfbde635449160af95ba2de9"
MODEL = source.MODEL
REVISION = source.REVISION
HORIZONS = 4
TRAIN_STEPS = source.MAX_STEPS
STAGE_STEPS = int(TRAIN_STEPS * source.STAGE_FRACTION)
GLOBAL_BATCH = source.GLOBAL_BATCH
PROBE_STEPS = source.PROBE_STEPS
TRAIN_HOURS = source.TRAIN_HOURS
PROBE_HOURS = 1
EVAL_HOURS = 12
RESERVATION_CAP_GPU_HOURS = 2 * (PROBE_HOURS * 4 + TRAIN_HOURS * 4 + EVAL_HOURS * 8)
CHECKPOINT_INTERVAL = TRAIN_STEPS // source.CHECKPOINT_PARTS
BACKEND = "flex_attention"
VAL_ROWS = 32
EVAL_COUNT = 100
REPEATS = 3
WARMUP_REQUESTS = 2
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.8
EAGLE_MODEL = "AngelSlim/Qwen3-1.7B_eagle3"
EAGLE_REVISION = "94441b48acc5804677ae12259617c83323b543a9"
AUDIT_COUNT = 2
AUDIT_PREFIX_TOKENS = 64
AUDIT_NEW_TOKENS = 16
GENERATION = source.GENERATION
SEED = source.SEED


@dataclass(frozen=True)
class Experiment:
    name: str
    arm: str
    mode: str | None = None
    num_gpus: int = 1


experiment_configs = [Experiment("qwen3_sps_mtp", "train", num_gpus=4)] + [
    Experiment(f"{arm}-{mode}", arm, mode)
    for mode in GENERATION
    for arm in ("ar", "eagle3", "sps", "sps_mtp")
]


def collect_experiments(phase, pattern=None):
    return [
        x
        for x in experiment_configs
        if (x.arm == "train") == (phase in {"probe", "train"})
        and (pattern is None or pattern in x.name)
    ]
