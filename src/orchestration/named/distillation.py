"""DFlash and Qwen block drafter launch plans."""

from orchestration.named.standalone import job, load


def dflash(kd):
    module = load(
        "workloads.distillation.dflash_kd"
        if kd
        else "workloads.distillation.dflash_repro"
    )
    hard = load("workloads.distillation.dflash_repro_train")
    training = (hard.reference, hard.prepared, hard, module.training)
    eval_settings = (*training, module.evaluation)
    for phase in ("probe", "train", "eval"):
        yield job(
            module,
            phase,
            module.training.EXPERIMENT,
            "--phase",
            phase,
            config=eval_settings if phase == "eval" else training,
        )
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
            config=eval_settings,
        )


def qwen4b():
    module = load("workloads.distillation.qwen4b_drafter")
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
                        output_dir=module.phase_output(phase, run.name, dataset),
                        config=(
                            module.training.reference,
                            module.prepared,
                            module.training,
                        ),
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
                    config=(
                        module.training.reference,
                        module.prepared,
                        module.training,
                        module.evaluation if phase == "eval" else module.training,
                    ),
                    output_dir=module.phase_output(
                        phase, module.names(phase, dataset)[0], dataset
                    ),
                    parameters={
                        "conditions": [
                            vars(run) for run in module.training.experiment_configs
                        ]
                    },
                )


def preview_drafter(plan, module, staged, launch):
    """Resolve only identity fields that have prepared inputs on this host."""
    if module.__name__.endswith(".qwen4b_drafter"):
        dataset = plan.args[plan.args.index("--dataset") + 1]
        training = module.training
        required = (
            training.DATA / "manifest.json"
            if dataset == training.DEFAULT_DATASET
            else training.selection_path()
        )
        if required.is_file():
            name = (
                plan.args[-1]
                if "--filter" in plan.args
                else module.names(plan.phase, dataset)[0]
            )
            return {"payload": module.payload(plan.phase, name, staged, dataset)}
        distributed = []
    else:
        hard = load("workloads.distillation.dflash_repro_train")
        required = hard.previous.selection_path()
        if required.is_file() or plan.phase != "eval" or "--step" in plan.args:
            step = None
            if plan.phase == "eval":
                step = (
                    int(plan.args[-1])
                    if "--step" in plan.args
                    else hard.total_steps(len(hard.previous.selection()["train"]))
                )
            return {
                "payload": module.payload(plan.phase, step, staged),
                "output_dir": module.training.EXPERIMENT.output(plan.phase, step),
            }
        distributed = (
            [
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={module.training.EXPERIMENT.num_gpus}",
            ]
            if plan.phase != "eval"
            else []
        )
    # Each worker resolves the final step or single evaluation arm at runtime.
    # Its job description and exact artifact ID depend on the missing file.
    return {
        "worker_command": [
            str(launch.PYTHON),
            *distributed,
            plan.script,
            *plan.args,
            "--worker",
        ],
        "pending_input": str(required),
        "pending_fields": ["job description", "artifact identity"],
    }
