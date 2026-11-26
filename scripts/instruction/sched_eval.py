"""Evaluation entry point for the sched instruction experiment."""
from functools import partial

from scripts import sched_common as sc, sched_config as c
from scripts.instruction import rerun_eval

evaluate = partial(rerun_eval.evaluate, sc, c)
