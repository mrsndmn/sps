"""Executable definitions for serving benchmark."""

from dataclasses import dataclass

from . import vllm_sps_config as c


@dataclass(kw_only=True)
class FamilyConfig:
    AUDIT_NAME: str = "vllm-greedy-audit"
    AUDIT_HOURS: int = 2
    MAX_ACTIVE_JOBS: int = 2


family_config = FamilyConfig()


experiment_configs = [
    c.Experiment(family_config.AUDIT_NAME, "audit")
] + c.collect_experiments("eval")
