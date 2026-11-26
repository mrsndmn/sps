"""Executable design for shared-backbone Qwen3 SPS-MTP and vLLM comparison."""

from dataclasses import dataclass, field
from pathlib import Path

from . import instruct_config as source
from .family_arguments import configure, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    ROOT: Path = Path(
        "/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/sps-artifacts/vllm-sps-qwen3"
    )
    SOURCE_SIGNATURE: str = (
        "761505551994ceec0a907401b3ee875ffea744b69bc94c7007b23c58181b466b"
    )
    PARENT_SIGNATURE: str = (
        "7408b3ccf0455895e306d0d5272e2fad41d3e70ae8c6fbb5612210d6d18347ae"
    )
    PARENT_STEP: int = 512
    VLLM_BASE: str = "3fe0d44219471ee5b178036a953b6f0c0c2bf5f6"
    VLLM_COMMIT: str = "65ff372c9f6fbdeebfbde635449160af95ba2de9"
    MODEL: str = source.MODEL
    REVISION: str = source.REVISION
    HORIZONS: int = 4
    TRAIN_STEPS: int = source.MAX_STEPS
    GLOBAL_BATCH: int = source.GLOBAL_BATCH
    PROBE_STEPS: int = source.PROBE_STEPS
    TRAIN_HOURS: int = source.TRAIN_HOURS
    PROBE_HOURS: int = 1
    EVAL_HOURS: int = 12
    BACKEND: str = "flex_attention"
    VAL_ROWS: int = 32
    EVAL_COUNT: int = 100
    REPEATS: int = 3
    WARMUP_REQUESTS: int = 2
    MAX_MODEL_LEN: int = 8192
    GPU_MEMORY_UTILIZATION: float = 0.8
    EAGLE_MODEL: str = "AngelSlim/Qwen3-1.7B_eagle3"
    EAGLE_REVISION: str = "94441b48acc5804677ae12259617c83323b543a9"
    AUDIT_COUNT: int = 2
    AUDIT_PREFIX_TOKENS: int = 64
    AUDIT_NEW_TOKENS: int = 16
    GENERATION: dict[str, dict] = field(default_factory=lambda: source.GENERATION)
    SEED: int = source.SEED

    @property
    def SOURCE_ROOT(self) -> Path:
        return self.ROOT.parent / "ar-sps-qwen3-1.7b-instruct"

    @property
    def PARENT(self) -> Path:
        return (
            self.ROOT.parent
            / "qwen3-instruct-cosine/runs/out/qwen3_instruct_staged_sps_ce/final.pt"
        )

    @property
    def VLLM_WORKTREE(self) -> Path:
        return self.ROOT.parent.parent / "vllm-sps"

    @property
    def STAGE_STEPS(self) -> int:
        return int(self.TRAIN_STEPS * source.STAGE_FRACTION)

    @property
    def RESERVATION_CAP_GPU_HOURS(self) -> int:
        return 2 * (self.PROBE_HOURS * 4 + self.TRAIN_HOURS * 4 + self.EVAL_HOURS * 8)

    @property
    def CHECKPOINT_INTERVAL(self) -> int:
        return self.TRAIN_STEPS // source.CHECKPOINT_PARTS


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)


@dataclass(frozen=True)
class Experiment:
    name: str
    arm: str
    mode: str | None = None
    num_gpus: int = 1


experiment_configs = [Experiment("qwen3_sps_mtp", "train", num_gpus=4)] + [
    Experiment(f"{arm}-{mode}", arm, mode)
    for mode in family_config.GENERATION
    for arm in ("ar", "eagle3", "sps", "sps_mtp")
]


def collect_experiments(phase, pattern=None):
    return [
        x
        for x in experiment_configs
        if (x.arm == "train") == (phase in {"probe", "train"})
        and (pattern is None or pattern in x.name)
    ]
