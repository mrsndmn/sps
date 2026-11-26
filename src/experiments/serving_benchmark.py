"""Executable definitions for serving benchmark."""

from . import vllm_sps_config as c

AUDIT_NAME = "vllm-greedy-audit"

AUDIT_HOURS = 2

MAX_ACTIVE_JOBS = 2

experiment_configs = [c.Experiment(AUDIT_NAME, "audit")] + c.collect_experiments("eval")
