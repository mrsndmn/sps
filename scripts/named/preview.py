"""Preview specialized submissions independently of old completion markers."""

from importlib import import_module
import json
import shlex
import subprocess
import sys

from scripts import mlspace_launch as launch


def preview(plan):
    module = import_module(plan.module)
    short = module.__name__.rsplit(".", 1)[-1]
    args = plan.args
    staged = launch.STAGING_ROOT / launch.git("rev-parse", "HEAD")
    author = ("d.tarasov", "mrsndmn")
    if short in {"dflash_repro", "dflash_kd", "qwen4b_drafter"}:
        from scripts.named.distillation import preview_drafter

        filled = preview_drafter(plan, module, staged, launch)
        print(json.dumps({"settings": plan.settings, **filled}, default=str, indent=2))
        return 0
    if short == "base50_run":
        body = module.payload(module.c.experiment_configs[0], staged)
    elif short in {"sched_run", "extend_run", "ifeval_run"}:
        name = args[args.index("--filter") + 1]
        (run,) = [r for r in launch.collect_experiments() if r.name == name]
        if short == "ifeval_run":
            body = module.payload(run, staged, *author, args[-1])
        else:
            workload = args[-1] if "--workload" in args else None
            body = module.payload(args[1], plan.phase, run, workload, staged, *author)
    elif short == "vllm_sps_run":
        (run,) = [r for r in module.c.experiment_configs if r.name == plan.name]
        body = module.payload(plan.phase, run, staged, *author)
    elif short == "qwen_adapter_run" and plan.phase == "eval":
        try:
            arms = [module.winner(launch.git("rev-parse", "HEAD"))]
            selection = "validated winner"
        except (FileNotFoundError, ValueError) as error:
            arms = module.experiment_configs
            selection = f"Winner pending valid training artifacts: {error}"
        print(
            json.dumps(
                dict(
                    settings=plan.settings,
                    selection=selection,
                    candidate_payloads=[
                        module.payload("eval", arm, staged) for arm in arms
                    ],
                ),
                default=str,
                indent=2,
            )
        )
        return 0
    elif short.startswith("real_readout_"):
        body = readout_payload(module, plan, staged, author)
    else:
        print(
            json.dumps({"settings": plan.settings}, default=str, indent=2), flush=True
        )
        return subprocess.call([sys.executable, "-m", plan.module, *args, "--dry"])
    print(json.dumps(dict(settings=plan.settings, payload=body), default=str, indent=2))
    return 0


def readout_payload(module, plan, staged, author):
    # Archived evaluator identities hash the complete module. Keep their
    # original entry points byte-identical when exposing them in the catalog.
    short = module.__name__.rsplit(".", 1)[-1]
    task = short.removeprefix("real_readout_")
    if task == "run":
        run = module.ARM
        hours = module.c.PROBE_HOURS if plan.phase == "probe" else module.c.TRAIN_HOURS
        cmd = [
            "timeout",
            f"{hours}h",
            str(launch.PYTHON),
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={run.num_gpus}",
            "-m",
            "scripts.real_readout_train",
            plan.phase,
        ]
        desc = f"sps-real-readout {plan.phase} {run.name}"
        body = launch.payload("train", run.name, staged, *author)
        body.update(instance_type=launch.INSTANCE_TYPES_BY_NUM_GPUS[run.num_gpus])
        body["env_variables"].update(HF_HUB_DISABLE_XET="1")
    else:
        (run,) = [
            r
            for r in module.experiment_configs
            if (r.name if plan.args[0] == "--filter" else r.arm) == plan.args[1]
        ]
        descriptions = {
            "eval": "sps-real-readout-eval",
            "gsm8k": "sps-readout-gsm8k",
            "humaneval": "sps-readout-humaneval-plus",
            "mmlu": "sps-readout-full-mmlu",
            "serving": "sps-real-readout-serving",
        }
        if task == "gsm8k":
            hours = module.HOURS[run.mode]
        elif task in {"mmlu", "humaneval"}:
            hours = module.HOURS
        else:
            hours = (module.ic if task == "eval" else module.c).EVAL_HOURS
        desc = f"{descriptions[task]} {run.name}"
        cmd = [
            "timeout",
            f"{hours}h",
            str(launch.PYTHON),
            "-m",
            "scripts." + short,
            "--worker",
            *plan.args,
        ]
        body = launch.payload("eval", run.name, staged, *author)
    body.update(
        job_desc=desc + f" #{author[0]} @{author[1]}",
        script=f"cd {shlex.quote(str(staged))} && " + shlex.join(cmd),
    )
    return body
