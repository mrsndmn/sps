"""DFlash and Qwen block drafter launch plans."""

from experiment_defs.standalone import job, load


def dflash(kd):
    module = load("dflash_kd" if kd else "dflash_repro")
    for phase in ("probe", "train", "eval"):
        yield job(
            module,
            phase,
            module.training.EXPERIMENT,
            "--phase",
            phase,
            config=module.training,
        )
    hard = load("dflash_repro_train")
    for step in hard.MILESTONES:
        yield job(
            module,
            "eval",
            module.training.EXPERIMENT,
            "--phase",
            "eval",
            "--step",
            step,
            name=f"{module.training.EXPERIMENT.name}-step-{step}",
            config=module.training,
        )


def qwen4b():
    module = load("qwen4b_drafter")
    for dataset in module.training.DATASETS:
        for phase in ("probe", "train", "eval"):
            # The evaluator derives its immutable artifact ID from the selected
            # corpus. Defer that lookup until the user launches this plan.
            if phase == "train":
                for run in module.training.experiment_configs:
                    yield job(
                        module,
                        phase,
                        run,
                        "--dataset",
                        dataset,
                        "--phase",
                        phase,
                        "--filter",
                        run.name,
                        name=f"{dataset}-{run.name}",
                        config=module.training,
                    )
            else:
                yield job(
                    module,
                    phase,
                    dataset,
                    "--dataset",
                    dataset,
                    "--phase",
                    phase,
                    config=module.evaluation if phase == "eval" else module.training,
                )
