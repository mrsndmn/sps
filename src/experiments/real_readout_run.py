"""Executable definitions for real readout run."""

from . import vllm_sps_config as c

ROOT = c.ROOT.parent / "qwen3-instruct-real-readout-mtp"

PARENT = c.PARENT.parents[1] / "qwen3_instruct_continued_ar_ce" / "final.pt"

experiment_configs = [
    c.Experiment(
        "qwen3_real_readout_mtp", "train", num_gpus=c.experiment_configs[0].num_gpus
    )
]

ARM = experiment_configs[0]

OUT = ROOT / "runs/out" / ARM.name

LEGACY_HORIZONS = 4
