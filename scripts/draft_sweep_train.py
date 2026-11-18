"""Train the nine-position SPS checkpoint shared by the draft sweep."""

import argparse
import hashlib
import json
import sys
from functools import partial
from pathlib import Path
from unittest.mock import patch

from scripts import vllm_sps_common as common
from scripts import vllm_sps_config as c

ROOT = c.ROOT.parent / "vllm-sps-draft-sweep"
HORIZONS = 9
EXPERIMENT = c.experiment_configs[0]


def signature():
    repo = Path(__file__).resolve().parents[1]
    sources = (
        "scripts/draft_sweep_train.py",
        "scripts/vllm_sps_train.py",
        "scripts/vllm_sps_model.py",
        "scripts/vllm_sps_common.py",
        "scripts/vllm_sps_config.py",
        "scripts/instruct_common.py",
        "scripts/instruct_config.py",
        "src/modeling/models/sps/pretrained.py",
    )
    hashes = {name: common.file_hash(repo / name) for name in sources}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def configure():
    c.ROOT = ROOT
    c.HORIZONS = HORIZONS
    common.signature = signature


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
    from scripts import vllm_sps_train as trainer

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


if __name__ == "__main__":
    main()
