"""Parallel IFEval design; reuse the existing instruct experiment registry."""

from dataclasses import dataclass


@dataclass(kw_only=True)
class FamilyConfig:
    SOURCE_SUITE: str = "ar-sps-qwen3-1.7b-instruct"
    ARTIFACT_NAME: str = "qwen3-instruct-ifeval"
    MODES: tuple[str, ...] = ("nonthinking", "thinking")
    EVAL_HOURS: int = 12
    STARTUP_SECONDS: int = 300
    GPU_HOURS: int = 125


family_config = FamilyConfig()


def launches(family):
    from .instruct_config import experiment_configs
    from .launch import filtered

    yield from filtered(
        experiment_configs,
        "eval",
        "scripts/instruction/ifeval_run.py",
        options=("mode",),
        required=("mode",),
    )
