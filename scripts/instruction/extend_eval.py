"""Evaluation entry point for the extend instruction experiment."""
from functools import partial

from scripts import extend_common as sc, extend_config as c
from scripts.instruction import rerun_eval

evaluate = partial(rerun_eval.evaluate, sc, c)
