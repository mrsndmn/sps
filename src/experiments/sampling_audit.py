"""Executable definitions for sampling audit."""

from dataclasses import dataclass


@dataclass(kw_only=True)
class FamilyConfig:
    SAMPLES: int = 32768
    SEED: int = 9147
    PREFIX_LENGTHS: tuple[int, ...] = (32, 127, 511, 1023)
    OUTPUT_LENGTHS: tuple[int, ...] = (1, 2, 3, 4, 5, 32, 128)
    KERNEL_CASES: tuple[tuple, ...] = (
        ("uniform", (1, 1, 1, 1, 1, 1, 1, 1), 1.0, 1.0, 8, (0, 1, 2)),
        ("skewed", (40, 20, 15, 10, 7, 4, 3, 1), 1.0, 1.0, 8, (0, 4, 7)),
        ("thinking", (40, 20, 15, 10, 7, 4, 3, 1), 0.6, 0.95, 8, (0, 1, 7)),
        ("nonthinking", (40, 20, 15, 10, 7, 4, 3, 1), 0.7, 0.8, 8, (0, 1, 7)),
        ("topk", (40, 20, 15, 10, 7, 4, 3, 1), 0.7, 1.0, 2, (0, 1, 7)),
        ("point_mass", (1, 0, 0, 0, 0, 0, 0, 0), 1.0, 1.0, 8, (0, 0, 1)),
    )


family_config = FamilyConfig()


@dataclass(frozen=True)
class Experiment:
    name: str = "kernel-and-cache"
    num_gpus: int = 1
    hours: int = 4


experiment_configs = [Experiment()]


def launches(family):
    from importlib import import_module
    from .launch import Launch

    for study in ("audit", "trace", "precision", "cache_trace", "backbone_precision"):
        module = import_module(f"experiments.sampling_{study}")
        for job in getattr(module, "job_configs", module.experiment_configs):
            yield Launch(job.name, "eval", f"scripts/serving/sampling_{study}.py", ("--filter", job.name))
