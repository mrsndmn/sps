"""Paired Qwen3.5 base and Unloop response-cap experiment."""

from dataclasses import dataclass

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
CAPS = Probe().caps
BATCH = 32
EVAL_HOURS = 4

RUNTIME_ENV = {"HF_HUB_OFFLINE": "1", "VLLM_USE_V2_MODEL_RUNNER": "0"}
