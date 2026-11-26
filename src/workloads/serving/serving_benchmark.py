"""Single-GPU native vLLM gate and benchmark using the isolated serving runtime."""
from experiments.launch import matching_runs
from experiments.family_cli import worker_child
from orchestration.named.publication import work_path

import argparse
import base64
import csv
import hashlib
import json
import os
import shlex
import signal
import subprocess
import time
from functools import partial
from pathlib import Path
from unittest.mock import patch

from orchestration import mlspace_launch as launch
from workloads.serving import serving_runtime as runtime
from workloads.serving import vllm_sps_audit as native_audit
from workloads.serving import vllm_sps_common as common
from experiments import vllm_sps_config as c
from workloads.serving import vllm_sps_evaluate as evaluator
from workloads.serving.vllm_sps_run import reserve
from experiments.serving_benchmark import family_config as serving_benchmark_settings
from experiments.serving_benchmark import experiment_configs
AUDIT_NAME = serving_benchmark_settings.AUDIT_NAME
AUDIT_HOURS = serving_benchmark_settings.AUDIT_HOURS
MAX_ACTIVE_JOBS = serving_benchmark_settings.MAX_ACTIVE_JOBS


def protocol_signature():
    inputs = (
        evaluator.protocol_signature(),


    )
    return hashlib.sha256(json.dumps(inputs).encode()).hexdigest()


def root_for(exp):
    return work_path(c.family_config.ROOT / "runs/out" / exp.name)


def reference_path(audit_name=AUDIT_NAME):
    return work_path(c.family_config.ROOT / "runs/out" / audit_name) / "reference.json"


def verify_runtime():
    manifest = json.loads((runtime.ARTIFACTS / "prepared.json").read_text())
    installed = json.loads((runtime.RUNTIME / "sps-runtime.json").read_text())
    if manifest != installed or any(installed.get(key) != value for key, value in runtime.identity().items()):
        raise RuntimeError("Runtime provenance mismatch")
    site = Path(manifest["site_packages"])
    if (
        not site.is_relative_to(runtime.RUNTIME)
        or not (runtime.RUNTIME / "bin/python").is_file()
    ):
        raise RuntimeError("Runtime paths do not match the isolated environment")
    for name, expected in manifest["overlay_sha256"].items():
        if common.file_hash(site / name) != expected:
            raise RuntimeError(f"Runtime overlay changed: {name}")
    if set(manifest["overlay_sha256"]) != set(runtime.OVERLAY):
        raise RuntimeError("Incomplete runtime overlay manifest")
    records = list(site.glob("vllm-*.dist-info/RECORD"))
    if len(records) != 1:
        raise RuntimeError("Ambiguous vLLM installation")
    direct = json.loads((records[0].parent / "direct_url.json").read_text())
    if direct["url"] != runtime.WHEEL:
        raise RuntimeError("Unrelated vLLM wheel installation")
    extension = "vllm/_C_stable_libtorch.abi3.so"
    with records[0].open() as stream:
        entries = {row[0]: row[1] for row in csv.reader(stream)}
    expected = entries.get(extension, "")
    actual = (
        base64.urlsafe_b64encode(bytes.fromhex(common.file_hash(site / extension)))
        .decode()
        .rstrip("=")
    )
    if expected != "sha256=" + actual:
        raise RuntimeError("Compiled vLLM extension changed")
    return manifest


def trained():
    return common.trained()


def audit_protocol(audit_name=AUDIT_NAME):
    root = work_path(c.family_config.ROOT / "runs/out" / audit_name)
    return json.loads((root / "serving-protocol.json").read_text())["protocol_signature"]


def reference_tokens(checkpoint, *, protocol=None, audit_name=AUDIT_NAME):
    data = json.loads(reference_path(audit_name).read_text())
    if (
        data["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]
        or data["protocol_signature"] != (protocol or protocol_signature())
    ):
        raise RuntimeError("Greedy oracle provenance mismatch")
    samples = data["samples"]
    if len(samples) != c.family_config.AUDIT_COUNT or any(
        not x["prompt_token_ids"]
        or not 0 < len(x["expected_token_ids"]) <= c.family_config.AUDIT_NEW_TOKENS
        for x in samples
    ):
        raise RuntimeError("Incomplete greedy oracle")
    return samples


def make_reference(audit_name=AUDIT_NAME):
    import torch
    import transformers

    checkpoint = trained()
    samples = native_audit.reference(checkpoint)
    launch.store_json(
        reference_path(audit_name),
        {
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "protocol_signature": protocol_signature(),
            "samples": samples,
        },
    )


def audit_arm(arm, audit_name=AUDIT_NAME):
    from workloads.serving.vllm_sps_evaluate import check_runtime, engine_arguments

    torch, vllm = check_runtime()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Audit requires exactly one visible GPU")
    checkpoint = trained()
    refs = reference_tokens(checkpoint, audit_name=audit_name)
    exp = c.Experiment(arm, arm)
    llm = vllm.LLM(**engine_arguments(exp, checkpoint["export"], c.family_config.EAGLE_REVISION))
    samples = native_audit.verify(llm, refs)
    launch.store_json(
        work_path(c.family_config.ROOT / "runs/out" / audit_name) / f"{arm}.json", {"samples": samples}
    )
    if not all(x["equal"] for x in samples):
        raise RuntimeError(f"{arm} native vLLM greedy parity failed")


def evaluate(exp):
    from transformers import AutoTokenizer

    from workloads.serving.vllm_sps_evaluate import evaluate as original

    check_gate(verify_runtime())
    # Keep the numerical oracle in the exact Torch/Transformers training
    # environment. Inference dependencies may contain a newer HF implementation.
    loader = partial(serving_tokenizer, AutoTokenizer.from_pretrained)
    with (
        patch.object(native_audit, "reference", reference_tokens),
        patch.object(AutoTokenizer, "from_pretrained", side_effect=loader),
    ):
        original(exp)


def serving_tokenizer(loader, *args, **kwargs):
    """Preserve token-list chat prompts across HF 4/5 return-dict defaults."""
    tokenizer = loader(*args, **kwargs)
    tokenizer.apply_chat_template = partial(
        tokenizer.apply_chat_template, return_dict=False
    )
    return tokenizer


def child(command, env, hours):
    process = subprocess.Popen(worker_child(list(map(str, command))), env=env, start_new_session=True)
    try:
        code = process.wait(timeout=hours * 3600)
        if code:
            raise RuntimeError(f"Serving worker exited {code}")
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise


def worker(exp):
    manifest = verify_runtime()
    if completed(exp, manifest):
        return
    # Context is durable before the original evaluator publishes its terminal
    # marker, so even a worker restart cannot lose the serving provenance.
    launch.store_json(root_for(exp) / "serving-runtime.json", manifest)
    launch.store_json(
        root_for(exp) / "serving-protocol.json",
        {
            "protocol_signature": protocol_signature(),
        },
    )
    env = runtime.runtime_environment()
    env.update(SPS_VLLM_SOURCE=manifest["site_packages"], VLLM_USE_V2_MODEL_RUNNER="0")
    python = runtime.RUNTIME / "bin/python"
    deadline = (
        time.monotonic() + (AUDIT_HOURS if exp.arm == "audit" else c.family_config.EVAL_HOURS) * 3600
    )

    def stage(command, settings):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Serving job exceeded its wall-clock budget")
        child(command, settings, remaining / 3600)

    if exp.arm == "audit":
        # Processes exit between the oracle and each serving engine, releasing
        # all GPU allocations before the next model starts.
        stage(
            [launch.PYTHON, "scripts/serving/serving_benchmark.py", "--reference-worker", "--filter", exp.name],
            os.environ.copy(),
        )
        for arm in ("sps", "sps_mtp"):
            stage(
                [python, "scripts/serving/serving_benchmark.py", "--audit-arm", arm, "--filter", exp.name],
                env,
            )
        launch.store_json(
            root_for(exp) / "evaluated.json",
            {
                "runtime_signature": runtime.signature(),
                "checkpoint_sha256": trained()["checkpoint_sha256"],
                "arms": ["sps", "sps_mtp"],
            },
        )
    else:
        check_gate(manifest)
        stage(
            [
                python,
                "scripts/serving/serving_benchmark.py",
                "--eval-worker",
                "--filter",
                exp.name,
            ],
            env,
        )


def selected(phase, name=None):
    rows = (exp for exp in experiment_configs if (exp.arm == "audit") == (phase == "audit"))
    return matching_runs(rows, name)


def payload(exp, staged, phase):
    body = launch.payload("eval", exp.name, staged, "d.tarasov", "mrsndmn")
    body.update(
        instance_type=launch.INSTANCE_TYPES_BY_NUM_GPUS[1],
        job_desc=f"sps-serving-benchmark {phase} {exp.name} #d.tarasov @mrsndmn",
    )
    body["script"] = f"cd {shlex.quote(str(staged))} && " + shlex.join(
        [
            str(launch.PYTHON),
            "scripts/serving/serving_benchmark.py",
            "--worker",
            "--phase",
            phase,
            "--filter",
            exp.name,
        ]
    )
    return body


def check_gate(manifest, *, protocol=None, audit_name=AUDIT_NAME):
    data = json.loads((work_path(c.family_config.ROOT / "runs/out" / audit_name) / "evaluated.json").read_text())
    if (
        data["runtime_signature"] != runtime.signature()
        or data["checkpoint_sha256"] != trained()["checkpoint_sha256"]
    ):
        raise RuntimeError("Greedy audit does not match this model and runtime")
    recorded = json.loads(
        (work_path(c.family_config.ROOT / "runs/out" / audit_name) / "serving-runtime.json").read_text()
    )
    if recorded != manifest:
        raise RuntimeError("Greedy audit runtime manifest changed")
    if audit_protocol(audit_name) != (protocol or protocol_signature()):
        raise RuntimeError("Greedy audit serving protocol changed")


def completed(exp, manifest, *, protocol=None):
    marker = root_for(exp) / "evaluated.json"
    if not marker.exists():
        return False
    if json.loads((root_for(exp) / "serving-runtime.json").read_text()) != manifest:
        raise RuntimeError("Completed evaluation used another runtime")
    context = json.loads((root_for(exp) / "serving-protocol.json").read_text())
    if context["protocol_signature"] != (protocol or protocol_signature()):
        raise RuntimeError("Completed evaluation used another serving protocol")
    if exp.arm != "audit":
        _, measured = common.evaluation_record(exp, trained()["checkpoint_sha256"])
        if protocol is None and measured.get("evaluation_protocol") != evaluator.protocol_signature():
            raise RuntimeError("Completed evaluation used another evaluator")
    check_gate(manifest, protocol=protocol, audit_name=exp.name if exp.arm == "audit" else AUDIT_NAME)
    return True


def submit(args):
    choices = selected(args.phase, args.filter)
    if not choices:
        raise ValueError("No matching experiments")
    commit = launch.git("rev-parse", "HEAD")
    staged = launch.STAGING_ROOT / commit
    if args.dry:
        for exp in choices:
            launch.print_preview(payload(exp, staged, args.phase), root_for(exp), staged, args.phase)
        return
    choices = [exp for exp in choices if launch.pending(root_for(exp))]
    if not choices:
        return
    manifest = verify_runtime()
    trained()
    if args.phase == "eval":
        check_gate(manifest)
    batch = launch.SubmissionBatch(commit)
    jobs = []
    deferred = []
    for exp in choices:
        if completed(exp, manifest):
            print("Already completed:", exp.name)
            continue
        body = payload(exp, staged, args.phase)
        job = batch.submit(body, "eval", exp.name, root_for(exp),
                           capacity=("sps-serving-benchmark ", MAX_ACTIVE_JOBS),
                           before_send=lambda: reserve("eval", exp))
        if job.get("deferred"):
            deferred.append(exp.name)
            continue
        jobs.append(job)
        print(json.dumps(job), flush=True)
    print("__MLS_JOBS_JSON__")
    print(json.dumps({"jobs": jobs, "deferred": deferred}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("audit", "eval"), default="audit")
    parser.add_argument("--filter")
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--reference-worker", action="store_true")
    parser.add_argument("--audit-arm", choices=("sps", "sps_mtp"))
    parser.add_argument("--eval-worker", action="store_true")
    args = parser.parse_args()
    if args.reference_worker or args.audit_arm:
        name = args.filter or AUDIT_NAME
        if not any(exp.arm == "audit" and exp.name == name for exp in experiment_configs):
            parser.error(f"Unknown audit run {name!r}")
        return make_reference(name) if args.reference_worker else audit_arm(args.audit_arm, name)
    if args.worker or args.eval_worker:
        choices = selected("eval" if args.eval_worker else args.phase, args.filter)
        if len(choices) != 1 or choices[0].name != args.filter:
            raise ValueError("Select one worker by exact experiment name")
        if args.eval_worker:
            return evaluate(choices[0])
        return worker(choices[0])
    launch.ARTIFACTS_ROOT = c.family_config.ROOT
    launch.RUN_ROOT = c.family_config.ROOT / "runs"
    if args.dry:
        return submit(args)
    with launch.submission_lock():
        return submit(args)
