"""Paired Qwen3.5 base and Unloop response-cap experiment."""

from dataclasses import dataclass, field

from .family_arguments import configure, configure_runs, export_fields
from .teacher_cap_probe import Probe


@dataclass(frozen=True)
class CapExperiment:
    name: str
    model: str
    revision: str


experiment_configs = {
    "original": CapExperiment(
        "original", "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"
    ),
    "unloop": CapExperiment(
        "unloop", "occ-ai/Qwen3.5-2B-Unloop", "3685440c358eef53ab2183f348b5ec9e9c37b12c"
    ),
}
experiment_configs = configure_runs(__name__, "experiment_configs", experiment_configs)


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    CAPS: tuple[int, ...] = Probe().caps
    OUTPUT_PARENT: tuple[str, str] = (
        "368b7c39020d6f4dda4a336ec92f8dc221e24ca187bf6539d9aec3dea2380e12",
        "185b483f6ef8312477e2f7895f7d879835cb129c6332c4aa3966d44317042406",
    )
    BATCH: int = 32
    EVAL_HOURS: int = 4
    RUNTIME_ENV: dict[str, str] = field(
        default_factory=lambda: {"HF_HUB_OFFLINE": "1", "VLLM_USE_V2_MODEL_RUNNER": "0"}
    )


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)
