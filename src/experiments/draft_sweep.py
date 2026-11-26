"""Executable definitions for draft sweep."""

from dataclasses import dataclass

from . import vllm_sps_config as c


@dataclass(kw_only=True)
class FamilyConfig:
    DRAFT_LENGTHS: tuple[int, ...] = (4, 8)
    EVAL_HOURS: int = 2
    AUDIT_TOKENS: int = 32

    @property
    def AUDIT_ARMS(self) -> dict[str, int]:
        return {"sps": 0, **{f"sps_mtp-{n}": n for n in self.DRAFT_LENGTHS}}


family_config = FamilyConfig()


@dataclass(frozen=True)
class Evaluation(c.Experiment):
    draft_tokens: int = 0


experiment_configs = [
    Evaluation(f"{arm}-{mode}", arm, mode)
    for mode in c.family_config.GENERATION
    for arm in ("ar", "sps")
] + [
    Evaluation(f"{arm}-{length}-{mode}", arm, mode, draft_tokens=length)
    for length in family_config.DRAFT_LENGTHS
    for mode in c.family_config.GENERATION
    for arm in ("eagle3", "sps_mtp")
]


def launches(family):
    from . import draft_sweep_train, serving_benchmark
    from .launch import filtered

    script = "scripts/serving/draft_sweep.py"
    for phase in ("train", "probe"):
        yield from filtered(
            (draft_sweep_train.EXPERIMENT,), phase, script, "--phase", phase
        )
    yield from filtered(
        (r for r in serving_benchmark.experiment_configs if r.arm == "audit"),
        "audit",
        script,
        "--phase",
        "audit",
    )
    yield from filtered(experiment_configs, "eval", script, "--phase", "eval")
