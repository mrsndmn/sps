"""Executable definitions for qwen adapter run."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from .environment import COMMON_ROOT as COMMON


@dataclass(kw_only=True)
class FamilyConfig:
    SUITE: str = "ar-sps-qwen3-1.7b"
    PARENT_SHA: str = "6ee490f323051d482770bfe0cabc938060caf95f672c39d6d73e117dbf77c1ae"
    SIGNATURE: str = "7dad73e5302bf23ef6bf64cb2b1ce9d7aa6633e7213b2db92fa79bb43d14b1c3"
    WHEEL: Path = COMMON / "toolchain/peft-0.17.1-py3-none-any.whl"
    WHEEL_SHA: str = "3d129d64def3d74779c32a080d2567e5f7b674e77d546e3585138216d903f99e"
    DATA_SHA: dict[str, str] = field(
        default_factory=lambda: dict(
            train="5b49b463f888c67072554d57972607a98dd10997d99d8ff09fa9586f59152796",
            val="51cbd70db36d60abbc0d7f7195743bee535be5edaf377e92b170323cff2da4e0",
            test="75322f02d3d55167129d4a4e8058321ae0f1bcb9982a82c5d9b36c9686b8ec58",
        )
    )
    START: int = 819
    END: int = 4096
    RANK: int = 16
    ALPHA: int = 32
    ATTENTION_TARGETS: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    PRED_LR: float = 1e-5

    @property
    def SOURCE(self) -> Path:
        return COMMON / self.SUITE / "runs"

    @property
    def PARENT(self) -> Path:
        return self.SOURCE / "out/qwen3_staged_sps_ce/stage_endpoint.pt"


family_config = FamilyConfig()


@dataclass(frozen=True)
class Arm:
    name: str
    dora: bool
    rs: bool
    lr: float


ATTENTION_ARMS = [
    Arm(f"{kind}-{scale}-{rate}", kind == "dora", scale == "rs", float(rate))
    for rate in ("3e-5", "1e-4")
    for kind in ("lora", "dora")
    for scale in ("standard", "rs")
]


@dataclass(frozen=True)
class AdapterSweep:
    output_name: str
    targets: tuple[str, ...]
    train_gpus: int
    arms: tuple[Arm, ...]


SWEEPS = {
    "attention": AdapterSweep(
        "qwen3-base-adapters", family_config.ATTENTION_TARGETS, 4, tuple(ATTENTION_ARMS)
    ),
    "attention-mlp": AdapterSweep(
        "qwen3-base-adapters-attention-mlp",
        family_config.ATTENTION_TARGETS + ("gate_proj", "up_proj", "down_proj"),
        8,
        (Arm("dora-attention-mlp-standard-1e-4", True, False, 1e-4),),
    ),
}


def arms(scope):
    return list(SWEEPS[scope].arms)


experiment_configs = arms(os.environ.get("SPS_ADAPTER_SCOPE", "attention"))
