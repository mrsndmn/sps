"""Train the nine-position SPS checkpoint shared by the draft sweep."""

import argparse
import sys
from functools import partial
from unittest.mock import patch

from experiments import vllm_sps_config as c
from experiments.draft_sweep_train import family_config as draft_sweep_train_settings
from experiments.draft_sweep_train import EXPERIMENT
ROOT = draft_sweep_train_settings.ROOT
LEGACY_HORIZONS = draft_sweep_train_settings.LEGACY_HORIZONS
HORIZONS = draft_sweep_train_settings.HORIZONS


def configure():
    c.family_config.ROOT = ROOT
    c.family_config.HORIZONS = HORIZONS


def training_optimizer(optimizer_for, model, frozen):
    # from_pretrained leaves the nested decoder in eval mode. The checkpointed
    # decoder layers only recompute activations when they are in training mode.
    model.train()
    optimizer = optimizer_for(model, frozen)
    if frozen:
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return optimizer


def main():
    from workloads.serving import vllm_sps_train as trainer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    configure()
    with (
        patch.object(
            trainer, "optimizer_for", partial(training_optimizer, trainer.optimizer_for)
        ),
        patch.object(sys, "argv", [__file__] + (["--probe"] if args.probe else [])),
    ):
        trainer.main()
