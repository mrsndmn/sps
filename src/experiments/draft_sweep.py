"""Executable definitions for draft sweep."""

from dataclasses import dataclass
from . import vllm_sps_config as c

DRAFT_LENGTHS = (4, 8)

EVAL_HOURS = 2

AUDIT_TOKENS = 32


@dataclass(frozen=True)
class Evaluation(c.Experiment):
    draft_tokens: int = 0


experiment_configs = [
    Evaluation(f"{arm}-{mode}", arm, mode)
    for mode in c.GENERATION
    for arm in ("ar", "sps")
] + [
    Evaluation(f"{arm}-{length}-{mode}", arm, mode, draft_tokens=length)
    for length in DRAFT_LENGTHS
    for mode in c.GENERATION
    for arm in ("eagle3", "sps_mtp")
]

AUDIT_ARMS = {"sps": 0, **{f"sps_mtp-{n}": n for n in DRAFT_LENGTHS}}
