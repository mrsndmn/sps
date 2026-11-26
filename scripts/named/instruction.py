"""Named checkpoint, schedule and workload choices for instruction-following runs."""

from scripts import instruct_config as source
from scripts.named.standalone import job, load


def schedules(extended):
    module = load("extend_run" if extended else "sched_run")
    config = module.c
    for variant in config.ROOT_NAMES:
        for run in source.experiment_configs:
            if not run.trainable:
                continue
            args = ("--variant", variant, "--filter", run.name)
            if variant == (str(config.TRAIN_STEPS) if extended else "cosine"):
                yield job(
                    module,
                    "train",
                    run,
                    *args,
                    "--phase",
                    "train",
                    config=(source, config),
                    name=f"{variant}-{run.name}",
                )
            for workload in config.WORKLOADS:
                yield job(
                    module,
                    "eval",
                    run,
                    *args,
                    "--phase",
                    "eval",
                    "--workload",
                    workload,
                    config=(source, config),
                    name=f"{variant}-{run.name}-{workload}",
                )


def ifeval():
    module = load("ifeval_run")
    for run in source.experiment_configs:
        for mode in module.c.MODES:
            yield job(
                module,
                "eval",
                run,
                "--filter",
                run.name,
                "--mode",
                mode,
                config=(source, module.c),
                name=module.arm_name(run, mode),
            )
