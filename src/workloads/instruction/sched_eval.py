"""Evaluation entry point for the sched instruction experiment."""
from functools import partial

from workloads.instruction import sched_common as sc
from experiments import sched_config as c
from workloads.instruction import rerun_eval

evaluate = partial(rerun_eval.evaluate, sc, c.family_config)
