"""Executable definitions for real readout humaneval."""

from dataclasses import dataclass, field
from pathlib import Path

from . import real_readout_run as train
from . import vllm_sps_config as c
from .family_arguments import configure, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    ROOT: Path = train.ROOT / "humaneval-plus"
    RELEASE: str = "v0.1.10"
    ARCHIVE_SHA: str = (
        "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101"
    )
    PROMPT_SHA: str = "2daaf2edc5a3dc6a68316a89800b61af7d534bca5f6230a5e6b6773411576c0a"
    REQUEST: str = (
        "Implement the Python function described below. Include its signature, all required "
        "imports and any helper functions. Return the complete solution in one Python code block."
    )
    SETTINGS: dict[str, dict] = field(
        default_factory=lambda: {
            mode: dict(settings, max_new_tokens=4096)
            for mode, settings in c.GENERATION.items()
        }
    )
    HOURS: int = 16

    @property
    def PROMPTS(self) -> Path:
        return self.ROOT / "input/prompts.jsonl"

    @property
    def URL(self) -> str:
        return f"https://github.com/evalplus/humanevalplus_release/releases/download/{self.RELEASE}/HumanEvalPlus.jsonl.gz"


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)


experiment_configs = [
    c.Experiment(f"{arm}-{mode}", arm, mode)
    for arm in ("A", "B")
    for mode in family_config.SETTINGS
]
