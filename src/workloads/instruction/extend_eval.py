"""Evaluation entry point for the extend instruction experiment."""
from functools import partial

from workloads.instruction import extend_common as sc
from experiments import extend_config as c
from workloads.instruction import rerun_eval

evaluate = partial(rerun_eval.evaluate, sc, c.family_config)
