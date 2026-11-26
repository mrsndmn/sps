"""Executable definitions for real readout run."""

from dataclasses import dataclass
from pathlib import Path

from . import vllm_sps_config as c


@dataclass(kw_only=True)
class FamilyConfig:
    ROOT: Path = c.family_config.ROOT.parent / "qwen3-instruct-real-readout-mtp"
    PARENT: Path = c.family_config.PARENT.parents[1] / "qwen3_instruct_continued_ar_ce" / "final.pt"
    LEGACY_HORIZONS: int = 4

    @property
    def OUT(self) -> Path:
        return self.ROOT / "runs/out" / experiment_configs[0].name


family_config = FamilyConfig()


@dataclass(frozen=True)
class Experiment(c.Experiment):
    probe_dir: str | None = None

    def output(self, phase):
        if phase == "probe":
            return family_config.ROOT / (self.probe_dir or f"probes/{self.name}")
        return family_config.ROOT / "runs/out" / self.name


experiment_configs = [
    Experiment(
        "qwen3_real_readout_mtp", "train", num_gpus=c.experiment_configs[0].num_gpus, probe_dir="probe"
    )
]


def launches(family):
    from importlib import import_module
    from .launch import Launch, filtered

    for phase in ("probe", "train"):
        yield from filtered(
            experiment_configs, phase, "scripts/readout/real_readout_run.py", phase
        )
    for task in ("eval", "mmlu", "gsm8k", "humaneval", "serving"):
        module = import_module(f"experiments.real_readout_{task}")
        for exp in module.experiment_configs:
            yield Launch(
                f"{task}-{exp.name}",
                "eval",
                f"scripts/readout/real_readout_{task}.py",
                ("--filter", exp.name),
            )
