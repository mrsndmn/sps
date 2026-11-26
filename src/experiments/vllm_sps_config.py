"""Executable design for shared-backbone Qwen3 SPS-MTP and vLLM comparison."""

from experiments.launch import matching_runs
from dataclasses import dataclass, field
from pathlib import Path

from . import instruct_config as source


@dataclass(kw_only=True)
class FamilyConfig:
    ROOT: Path = Path(
        "/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/sps-artifacts/vllm-sps-qwen3"
    )
    PARENT_STEP: int = 512
    VLLM_BASE: str = "3fe0d44219471ee5b178036a953b6f0c0c2bf5f6"
    VLLM_COMMIT: str = "65ff372c9f6fbdeebfbde635449160af95ba2de9"
    MODEL: str = source.family_config.MODEL
    REVISION: str = source.family_config.REVISION
    HORIZONS: int = 4
    TRAIN_STEPS: int = source.family_config.MAX_STEPS
    GLOBAL_BATCH: int = source.family_config.GLOBAL_BATCH
    PROBE_STEPS: int = source.family_config.PROBE_STEPS
    TRAIN_HOURS: int = source.family_config.TRAIN_HOURS
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
    GENERATION: dict[str, dict] = field(default_factory=lambda: source.family_config.GENERATION)
    SEED: int = source.family_config.SEED

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
        return int(self.TRAIN_STEPS * source.family_config.STAGE_FRACTION)

    EVAL_JOBS: int = 8

    @property
    def RESERVATION_CAP_GPU_HOURS(self) -> int:
        return 2 * (self.PROBE_HOURS * 4 + self.TRAIN_HOURS * 4 + self.EVAL_HOURS * self.EVAL_JOBS)

    @property
    def CHECKPOINT_INTERVAL(self) -> int:
        return self.TRAIN_STEPS // source.family_config.CHECKPOINT_PARTS


family_config = FamilyConfig()


@dataclass(frozen=True)
class Experiment:
    name: str
    arm: str
    mode: str | None = None
    num_gpus: int = 1

    def output(self, phase):
        return family_config.ROOT / ("sanity" if phase == "probe" else "runs") / "out" / self.name


experiment_configs = [Experiment("qwen3_sps_mtp", "train", num_gpus=4)] + [
    Experiment(f"{arm}-{mode}", arm, mode)
    for mode in family_config.GENERATION
    for arm in ("ar", "eagle3", "sps", "sps_mtp")
]


def collect_experiments(phase, pattern=None):
    rows = (exp for exp in experiment_configs if (exp.arm == "train") == (phase in {"probe", "train"}))
    return matching_runs(rows, pattern)


def launches(family):
    from .launch import filtered

    for phase in ("probe", "train"):
        yield from filtered(
            collect_experiments(phase),
            phase,
            "scripts/serving/vllm_sps_run.py",
            "--phase",
            phase,
        )
