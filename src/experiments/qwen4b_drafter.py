"""Protocol settings for qwen4b drafter."""

from dataclasses import dataclass, field


@dataclass(kw_only=True)
class FamilyConfig:
    DATASET_DIRECTORIES: dict[str, str] = field(default_factory=lambda: {
        "smoltalk-2048": "b0088cb54b3bd646",
        "smoltalk": "smoltalk",
        "dflash-100k": "dflash-100k",
    })
    PHASE_TIMEOUT: dict[str, str] = field(
        default_factory=lambda: {"probe": "40m", "train": "6h", "eval": "3h"}
    )


family_config = FamilyConfig()


def launches(family):
    from .qwen4b_drafter_train import experiment_configs
    from .launch import Launch, filtered

    script = "scripts/distillation/qwen4b_drafter.py"
    yield from filtered(
        experiment_configs, "train", script, "--phase", "train", options=("dataset",)
    )
    for name, phase in (("probe", "probe"), ("evaluation", "eval")):
        yield Launch(name, phase, script, ("--phase", phase), options=("dataset",))
