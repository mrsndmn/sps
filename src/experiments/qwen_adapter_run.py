"""Executable definitions for qwen adapter run."""

from dataclasses import dataclass
from .environment import COMMON_ROOT as COMMON

SUITE = "ar-sps-qwen3-1.7b"

SOURCE = COMMON / SUITE / "runs"

PARENT = SOURCE / "out/qwen3_staged_sps_ce/stage_endpoint.pt"

PARENT_SHA = "6ee490f323051d482770bfe0cabc938060caf95f672c39d6d73e117dbf77c1ae"

SIGNATURE = "7dad73e5302bf23ef6bf64cb2b1ce9d7aa6633e7213b2db92fa79bb43d14b1c3"

WHEEL = COMMON / "toolchain/peft-0.17.1-py3-none-any.whl"

WHEEL_SHA = "3d129d64def3d74779c32a080d2567e5f7b674e77d546e3585138216d903f99e"

DATA_SHA = dict(
    train="5b49b463f888c67072554d57972607a98dd10997d99d8ff09fa9586f59152796",
    val="51cbd70db36d60abbc0d7f7195743bee535be5edaf377e92b170323cff2da4e0",
    test="75322f02d3d55167129d4a4e8058321ae0f1bcb9982a82c5d9b36c9686b8ec58",
)

START, END = 819, 4096

RANK, ALPHA = 16, 32

ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")

PRED_LR = 1e-5


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
        "qwen3-base-adapters", ATTENTION_TARGETS, 4, tuple(ATTENTION_ARMS)
    ),
    "attention-mlp": AdapterSweep(
        "qwen3-base-adapters-attention-mlp",
        ATTENTION_TARGETS + ("gate_proj", "up_proj", "down_proj"),
        8,
        (Arm("dora-attention-mlp-standard-1e-4", True, False, 1e-4),),
    ),
}
