"""Launch the isolated DFlash-b16 reproduction probe, training, or evaluation."""

import argparse
from dataclasses import asdict, fields, replace
import sys

from experiments.dflash_repro import DFlashEvalArguments, ModelArguments
from orchestration.named.publication import submitting, work_path
import json
from pathlib import Path
import shlex
from unittest.mock import patch

from workloads.distillation import (
    dflash_repro_eval as evaluation,
    dflash_repro_train as training,
)
from orchestration import mlspace_launch as launch

from experiments.launch import run_filter


def marker(experiment, phase, step=None):
    filename = {
        "probe": "prepared.json",
        "train": "trained.json",
        "eval": "evaluated.json",
    }[phase]
    path = work_path(experiment.output(phase, step)) / filename
    if phase == "probe" and experiment.archived_eval_pattern and not path.is_file():
        archived = experiment.output_dir / filename
        if archived.is_file():
            design = json.loads(archived.read_text()).get("design")
            if isinstance(design, str) and design[:16] == experiment.output_dir.name:
                return archived
    return path


def worker_arguments(trainer, experiment, phase, step):
    train_args = trainer.TrainArguments(
        **{f.name: getattr(experiment, f.name) for f in fields(trainer.TrainArguments) if f.name != "result_dir"},
        result_dir=experiment.output("probe" if phase == "probe" else "train"),
    )
    if phase == "eval":
        return DFlashEvalArguments(
            **asdict(experiment.eval_protocol),
            **{f.name: getattr(experiment, f.name) for f in fields(ModelArguments)},
            name=experiment.name,
            checkpoint=trainer.checkpoint_path(step, experiment.output("train")),
            result=experiment.output(phase, step) / "evaluated.json",
            proposals=experiment.proposals,
            seed=experiment.seed,
            training_settings_sha256=evaluation.settings_digest(
                training.training_settings(train_args)
            ),
        )
    return train_args


def worker(trainer, evaluate, phase, argv):
    from transformers import HfArgumentParser

    schema = DFlashEvalArguments if phase == "eval" else trainer.TrainArguments
    (args,) = HfArgumentParser(schema).parse_args_into_dataclasses(
        argv, look_for_args_file=False
    )
    if phase == "eval":
        if work_path(args.result).is_file():
            return
        if not (args.checkpoint.parent / "trained.json").is_file():
            raise RuntimeError(
                "Complete DFlash training before evaluating any checkpoint"
            )
        return evaluate(replace(args, result=work_path(args.result)))
    if phase not in ("train", "probe"):
        raise ValueError(f"Unknown phase {phase}")
    filename = "prepared.json" if phase == "probe" else "trained.json"
    if not (work_path(args.result_dir) / filename).is_file():
        return trainer.train(experiment=args, probe=phase == "probe")


def payload(experiment, phase, step, staged, script, arguments):
    name = experiment.output(phase, step).name if phase == "eval" else experiment.name
    body = launch.payload(
        "eval" if phase == "eval" else "train", name, staged, "d.tarasov", "mrsndmn"
    )
    command = ["timeout", "--kill-after=60s", experiment.timeouts[phase], str(launch.PYTHON)]
    if phase != "eval":
        body["instance_type"] = launch.INSTANCE_TYPES_BY_NUM_GPUS[experiment.train_gpus]
        body["env_variables"]["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        command += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={experiment.train_gpus}",
        ]
    command += [script, "--worker", phase]
    for key, value in asdict(arguments).items():
        if value == []:
            continue
        command.append(f"--{key}")
        command.extend(map(str, value if isinstance(value, list) else [value]))
    body["script"] = f"cd {shlex.quote(str(staged))} && " + shlex.join(command)
    study = Path(script).stem.replace("_", "-")
    body["job_desc"] = (
        f"{study} {phase} {name} {experiment.output_dir.name} #d.tarasov @mrsndmn"
    )
    return body


def submit_phase(experiment, phase, step, body, commit, arguments):
    output = experiment.output(phase, step)
    if phase == "probe" and marker(experiment, phase).is_file():
        print(f"Skip completed probe {experiment.name}")
        return
    with submitting(output) as ready:
        if not ready:
            return
        if phase == "train" and not marker(experiment, "probe").is_file():
            raise RuntimeError("Complete the DFlash GPU probe before training")
        if phase == "eval":
            if not marker(experiment, "train").is_file():
                raise RuntimeError(
                    "Complete DFlash training before evaluating any checkpoint"
                )
            weights = arguments.checkpoint / "weights.safetensors"
            if not weights.is_file():
                raise FileNotFoundError(weights)
            evaluation.validate_checkpoint(arguments)
        name = output.name if phase == "eval" else experiment.name
        with (
            patch.object(launch, "ARTIFACTS_ROOT", experiment.output_dir),
            launch.submission_lock(),
        ):
            receipt = launch.SubmissionBatch(commit).submit(
                body,
                "prepare" if phase == "probe" else phase,
                name,
                output,
                settings=asdict(arguments),
            )
            print(json.dumps(receipt), flush=True)


def run(trainer, evaluate):
    if sys.argv[1:2] == ["--worker"]:
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        parser.add_argument("phase", choices=trainer.Config.phases)
        control, fields_argv = parser.parse_known_args(sys.argv[2:])
        return worker(trainer, evaluate, control.phase, fields_argv)
    parser = argparse.ArgumentParser(description=__doc__)
    configs = run_filter(parser, trainer.Config.experiment_configs())
    parser.add_argument("--phase", choices=trainer.Config.phases, default="probe")
    parser.add_argument("--step", type=int)
    modes = parser.add_mutually_exclusive_group()
    for flag in ("dry", "report"):
        modes.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    experiment = configs[args.filter]
    phase, step = args.phase, args.step
    if phase == "eval" and step not in experiment.milestones:
        trained = marker(experiment, "train")
        if args.dry and not trained.is_file():
            print(json.dumps({"pending_input": str(trained)}))
            return
        final = json.loads(trained.read_text())["steps"]
        if step is not None and step != final:
            parser.error("Choose a scheduled DFlash checkpoint")
        step = final
    if phase != "eval" and step is not None:
        parser.error("--step is only available for evaluation")
    result = marker(experiment, phase, step)
    if args.report:
        report = json.loads(result.read_text())
        print(json.dumps(report["summary"] if phase == "eval" else report, indent=2))
    else:
        commit = launch.git("rev-parse", "HEAD")
        staged = launch.STAGING_ROOT / commit
        arguments = worker_arguments(trainer, experiment, phase, step)
        body = payload(experiment, phase, step, staged, trainer.Config.script, arguments)
        if args.dry:
            launch.print_preview(
                body,
                experiment.output(phase, step),
                staged,
                phase,
                settings=asdict(arguments),
            )
        else:
            submit_phase(experiment, phase, step, body, commit, arguments)


def main():
    run(training, evaluation.evaluate)
