"""Longer training at the same scale as the pilot."""

from dataclasses import dataclass

from experiments.environment import COMMON_ROOT
from experiments.pilot import Model, Pilot, Training


@dataclass(frozen=True, kw_only=True)
class Pilot10B(Pilot):
    training: Training = Training(
        max_tokens=10_003_415_040,
        warmup_tokens=100_000_000,
        decay_tokens=1_000_000_000,
        eval_interval=250_000_000,
        save_interval=1_000_000_000,
        rolling_interval=250_000_000,
    )


experiment_configs = [
    Pilot10B(
        name="s_standard_10b",
        model=Model(),
        reference_name="s_standard_pilot_1b",
        output_dir=COMMON_ROOT / "10b/runs/out/s_standard_10b",
        eval_output_dir=COMMON_ROOT / "10b/runs/out/s_standard_10b/eval",
    ),
    Pilot10B(
        name="s_sps_w0_10b",
        model=Model(window=0),
        reference_name="s_sps_w0_pilot_1b",
        output_dir=COMMON_ROOT / "10b/runs/out/s_sps_w0_10b",
        eval_output_dir=COMMON_ROOT / "10b/runs/out/s_sps_w0_10b/eval",
    ),
]
