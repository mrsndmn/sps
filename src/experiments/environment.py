"""Shared storage paths and MLSpace defaults for experiment workers."""

from pathlib import Path

COMMON_ROOT = Path("/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/sps-artifacts")
STAGING_ROOT = COMMON_ROOT / "code"
HF_HOME = COMMON_ROOT / "hf"
PYTHON = Path("/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/sps/.venv/bin/python")
INSTANCE_TYPES_BY_NUM_GPUS = {1: "a100.1gpu", 4: "a100.4gpu"}
EVAL_NUM_GPUS = 1  # Project policy: evaluation workers always use one GPU.
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41"
