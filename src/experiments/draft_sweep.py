"""Executable definitions for draft sweep."""

from dataclasses import dataclass

from . import vllm_sps_config as c
from .family_arguments import configure, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    DRAFT_LENGTHS: tuple[int, ...] = (4, 8)
    EVAL_HOURS: int = 2
    AUDIT_TOKENS: int = 32

    @property
    def AUDIT_ARMS(self) -> dict[str, int]:
        return {"sps": 0, **{f"sps_mtp-{n}": n for n in self.DRAFT_LENGTHS}}


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)


@dataclass(frozen=True)
class Evaluation(c.Experiment):
    draft_tokens: int = 0


experiment_configs = [
    Evaluation(f"{arm}-{mode}", arm, mode)
    for mode in c.GENERATION
    for arm in ("ar", "sps")
] + [
    Evaluation(f"{arm}-{length}-{mode}", arm, mode, draft_tokens=length)
    for length in family_config.DRAFT_LENGTHS
    for mode in c.GENERATION
    for arm in ("eagle3", "sps_mtp")
]
