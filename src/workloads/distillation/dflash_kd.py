"""Submit and report the DFlash-b16 soft-distillation experiment."""

import argparse
from hashlib import sha256

from orchestration.design_compatibility import canonical_signature
import json
from pathlib import Path
import shlex
import tempfile

from workloads.distillation import dflash_kd_train as training, dflash_repro as runner
from workloads.distillation import dflash_repro_eval as evaluation
from workloads.distillation import dflash_repro_train as hard
from orchestration import mlspace_launch as launch


def evaluation_signature():
    return canonical_signature(sha256(evaluation.signature().encode() + Path(__file__).read_bytes()).hexdigest())


def arm_name(phase, step):
    if phase != "eval":
        return training.EXPERIMENT.name
    return f"gsm8k-step-{step:05d}-{evaluation_signature()[:12]}"


def marker(phase, step):
    if phase == "probe":
        return training.run_root() / "prepared.json"
    name = "trained.json" if phase == "train" else "evaluated.json"
    return training.run_root() / "runs" / "out" / arm_name(phase, step) / name


def payload(phase, step, staged):
    name = arm_name(phase, step)
    body = launch.payload("eval" if phase == "eval" else "train", name, staged, "d.tarasov", "mrsndmn")
    distributed = []
    if phase != "eval":
        gpus = training.EXPERIMENT.num_gpus
        body["instance_type"] = f"a100.{gpus}gpu"
        body["env_variables"]["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        distributed = ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={gpus}"]
    entry = ["scripts/distillation/dflash_kd.py", "--worker", "--phase", phase]
    if phase == "eval":
        entry += ["--step", str(step)]
    command = ["timeout", "--kill-after=60s", runner.TIMEOUTS[phase], str(launch.PYTHON),
               *distributed, *entry]
    body["script"] = f"cd {shlex.quote(str(staged))} && " + shlex.join(command)
    body["job_desc"] = f"dflash-kd {phase} {name} {training.signature()[:16]} #d.tarasov @mrsndmn"
    return body


def evaluate(step, result):
    # Reuse the exact hard-run evaluation protocol without changing its archived identity.
    result.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=result.parent) as tmp:
        raw = Path(tmp) / "raw.json"
        evaluation.evaluate(training.checkpoint_path(step), raw)
        report = json.loads(raw.read_text())
        report.update(design=training.signature(), evaluation=evaluation_signature(),
                      hard_design=hard.signature(), objective="soft-kl")
        launch.store_json(result, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(runner.TIMEOUTS), default="probe")
    parser.add_argument("--step", type=int)
    modes = parser.add_mutually_exclusive_group()
    for flag in ("dry", "worker", "report"):
        modes.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    phase, step = args.phase, args.step
    if phase == "eval":
        last = runner.final_step()
        step = last if step is None else step
        if step not in (*hard.MILESTONES, last):
            parser.error("Choose a scheduled DFlash checkpoint")
    elif step is not None:
        parser.error("--step is only available for evaluation")

    result = marker(phase, step)
    if args.report:
        report = json.loads(result.read_text())
        print(json.dumps(report["summary"] if phase == "eval" else report, indent=2))
    elif args.worker:
        if result.is_file():
            return
        if phase == "eval":
            evaluate(step, result)
        else:
            training.train(probe=phase == "probe")
    else:
        commit = launch.git("rev-parse", "HEAD")
        body = payload(phase, step, launch.STAGING_ROOT / commit)
        if args.dry:
            print(json.dumps(body, indent=2))
            return
        if phase == "train" and not marker("probe", None).is_file():
            raise RuntimeError("Complete the soft-KD GPU probe before training")
        if phase == "eval" and not (training.checkpoint_path(step) / "weights.safetensors").is_file():
            raise RuntimeError(f"Missing soft-KD checkpoint at step {step}")
        runner.submit_job(phase, arm_name(phase, step), training.run_root(), result, body, commit)
