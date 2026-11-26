"""Executable definitions for serving benchmark."""

from dataclasses import dataclass

from . import vllm_sps_config as c
from .family_arguments import configure, configure_runs, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    AUDIT_NAME: str = "vllm-greedy-audit"
    AUDIT_HOURS: int = 2
    MAX_ACTIVE_JOBS: int = 2


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)


experiment_configs = [
    c.Experiment(family_config.AUDIT_NAME, "audit")
] + c.collect_experiments("eval")
experiment_configs = configure_runs(__name__, "experiment_configs", experiment_configs)
