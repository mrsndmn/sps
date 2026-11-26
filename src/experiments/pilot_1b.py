"""Token-matched S-scale pilot runs."""

from dataclasses import dataclass

from experiments.environment import COMMON_ROOT
from experiments.pilot import Model, Pilot, Training


@dataclass(frozen=True, kw_only=True)
class Pilot1B(Pilot):
    training: Training = Training(
        max_tokens=1_000_341_504,
        warmup_tokens=10_000_000,
        decay_tokens=100_000_000,
        eval_interval=50_000_000,
        save_interval=250_000_000,
        rolling_interval=100_000_000,
    )


experiment_configs = [
    Pilot1B(
        name="s_standard_pilot_1b",
        model=Model(),
        output_dir=COMMON_ROOT / "runs/out/s_standard_pilot_1b",
        eval_output_dir=COMMON_ROOT / "runs/out/s_standard_pilot_1b/eval",
    ),
    Pilot1B(
        name="s_sps_w0_pilot_1b",
        model=Model(window=0),
        output_dir=COMMON_ROOT / "runs/out/s_sps_w0_pilot_1b",
        eval_output_dir=COMMON_ROOT / "runs/out/s_sps_w0_pilot_1b/eval",
    ),
]
