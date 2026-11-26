"""Run the original trainer or evaluator with a named Python configuration."""

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

import hydra
from hydra.core.config_store import ConfigStore
from orchestration.experiment_suite import RUN_ROOT, SANITY_ROOT, collect_experiments


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("phase", choices=("train", "eval"))
    parser.add_argument("name")
    parser.add_argument("--probe", action="store_true")
    args, hydra_args = parser.parse_known_args()
    (exp,) = (exp for exp in collect_experiments() if exp.name == args.name)
    config = exp.config(SANITY_ROOT if args.probe else RUN_ROOT, probe=args.probe)
    # Keep the project's Hydra logging/output policy when composing this entry.
    system_config = OmegaConf.load(
        Path(__file__).resolve().parents[3] / "conf/config.yaml"
    )
    config["hydra"] = system_config.hydra
    config["defaults"] = [
        {"override hydra/hydra_logging": "disabled"},
        {"override hydra/job_logging": "disabled"},
        "_self_",
    ]
    ConfigStore.instance().store(name="named_pilot", node=config)
    sys.argv[1:] = hydra_args
    if args.phase == "train":
        from training.train import main as run
    else:
        from modeling.evaluation.evaluate import main as run
    hydra.main(version_base=None, config_path=None, config_name="named_pilot")(
        run.__wrapped__
    )()
