"""Executable definitions for sampling trace."""

from dataclasses import dataclass

from .family_arguments import configure, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    ARMS: tuple[str, ...] = ("sps", "sps_four_slots", "sps_mtp")
    NATIVE_ARMS: tuple[str, ...] = (
        "native-fp32-one",
        "native-fp32-four",
        "native-bf16-one",
        "native-bf16-four",
    )


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)


@dataclass(frozen=True)
class Experiment:
    name: str = "divergence-distributions"
    num_gpus: int = 1
    hours: int = 2


experiment_configs = [Experiment()]
