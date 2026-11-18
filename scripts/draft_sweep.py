"""Run the Qwen3 vLLM draft-length sweep with isolated training and artifacts."""

import argparse
import csv
import hashlib
import json
import os
import shlex
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from unittest.mock import patch

from scripts import draft_sweep_train as training
from scripts import serving_benchmark as bench
from scripts import serving_graphs as graph
from scripts import serving_runtime as runtime
from scripts import vllm_sps_common as common
from scripts import vllm_sps_config as c
from scripts import vllm_sps_run as run

ROOT = training.ROOT
LAUNCHER = "scripts.draft_sweep"
DRAFT_LENGTHS = (4, 8)
HORIZONS = training.HORIZONS
PHASES = ("probe", "train", "audit", "eval")
EVAL_HOURS = 2
AUDIT_TOKENS = 32
TRAINER = training.EXPERIMENT
AUDIT = c.Experiment(bench.AUDIT_NAME, "audit")


@dataclass(frozen=True)
class Evaluation(c.Experiment):
    draft_tokens: int = 0


experiment_configs = [
    Evaluation(f"{arm}-{mode}", arm, mode)
    for mode in c.GENERATION
    for arm in ("ar", "sps")
] + [
    Evaluation(f"{arm}-{length}-{mode}", arm, mode, draft_tokens=length)
    for length in DRAFT_LENGTHS
    for mode in c.GENERATION
    for arm in ("eagle3", "sps_mtp")
]
AUDIT_ARMS = {"sps": 0, **{f"sps_mtp-{n}": n for n in DRAFT_LENGTHS}}


def protocol_signature():
    parts = (
        training.signature(),
        common.file_hash(Path(__file__)),
        common.file_hash(Path(bench.__file__)),
        common.file_hash(Path(graph.__file__)),
        runtime.signature(),
        graph.IMPLEMENTATION,
    )
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def configure():
    training.configure()
    c.EVAL_HOURS = EVAL_HOURS
    c.AUDIT_NEW_TOKENS = AUDIT_TOKENS
    c.experiment_configs = [TRAINER, *experiment_configs]
    c.RESERVATION_CAP_GPU_HOURS = 2 * (
        (c.PROBE_HOURS + c.TRAIN_HOURS) * TRAINER.num_gpus
        + (len(experiment_configs) + 1) * EVAL_HOURS
    )
    bench.protocol_signature = graph.protocol_signature = protocol_signature
    bench.verify_runtime = graph.verify_runtime
    bench.check_gate = check_gate
    graph.engine_arguments = engine_arguments


def engine_arguments(exp, export, eagle_revision):
    arm, _, length = exp.arm.partition("-")
    draft = getattr(exp, "draft_tokens", None)
    if draft is None:
        draft = int(length) if length else 0
    export = export_for(arm, draft, export)
    result = BASE_ENGINE_ARGUMENTS(replace(exp, arm=arm), export, eagle_revision)
    if draft:
        result["speculative_config"]["num_speculative_tokens"] = draft
    return result


BASE_ENGINE_ARGUMENTS = graph.engine_arguments
BASE_GATE = bench.check_gate


def export_for(arm, draft, export):
    if arm == "sps_mtp" and draft in DRAFT_LENGTHS[:-1]:
        return Path(export).with_name(f"hf-draft-{draft}")
    return Path(export)


def prepare_short_export(export, *, create=True):
    """Expose the same checkpoint at the short SPS width for genuine timing."""
    from safetensors.torch import load_file, save_file

    export = Path(export)
    config = json.loads((export / "config.json").read_text())
    key = "sps_prediction_embeddings"
    embeddings = load_file(export / "sps.safetensors")[key]
    if config["sps_horizons"] != HORIZONS or embeddings.shape[0] != HORIZONS:
        raise ValueError("Unexpected source export width")
    for length in DRAFT_LENGTHS[:-1]:
        width = length + 1
        target = export_for("sps_mtp", length, export)
        index_path = "model.safetensors.index.json"
        index = json.loads((export / index_path).read_text())
        index["metadata"]["total_size"] -= (
            embeddings[width:].numel() * embeddings.element_size()
        )
        narrowed = {key: embeddings[:width].clone()}
        if target.exists():
            if (
                json.loads((target / "config.json").read_text())
                != dict(config, sps_horizons=width)
                or json.loads((target / index_path).read_text()) != index
                or not load_file(target / "sps.safetensors")[key].equal(narrowed[key])
                or any(
                    (target / source.name).resolve() != source.resolve()
                    for source in export.iterdir()
                    if source.name not in {"config.json", "sps.safetensors", index_path}
                )
            ):
                raise RuntimeError(
                    "Existing short SPS export differs from the trained model"
                )
            continue
        if not create:
            raise RuntimeError(f"Missing short SPS export: {target}")
        with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
            staged = Path(temporary) / target.name
            staged.mkdir()
            for source in export.iterdir():
                if source.name not in {"config.json", "sps.safetensors", index_path}:
                    (staged / source.name).symlink_to(source.resolve())
            bench.launch.store_json(
                staged / "config.json", dict(config, sps_horizons=width)
            )
            bench.launch.store_json(staged / index_path, index)
            save_file(narrowed, staged / "sps.safetensors")
            staged.rename(target)


def check_gate(manifest):
    BASE_GATE(manifest)
    prepare_short_export(bench.trained()["export"], create=False)
    for arm in AUDIT_ARMS:
        graph.validate_evidence(bench.root_for(AUDIT) / f"graphs-{arm}.json")


def graph_job(function, argument, evidence, draft):
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Serving evaluations require exactly one GPU")
    sizes = list(range(1, draft + 2))
    with patch.dict(graph.COMPILATION, cudagraph_capture_sizes=sizes):
        graph.with_graph_engine(function, argument, evidence)


def evaluate(exp):
    import vllm

    root = bench.root_for(exp)
    needed = c.WARMUP_REQUESTS + (c.AUDIT_COUNT if exp.arm == "sps_mtp" else 0)
    original = vllm.LLM

    class MeasuredLLM(original):
        untimed_requests = 0

        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            self.untimed_requests += 1
            if exp.draft_tokens and self.untimed_requests == needed:
                counters = [
                    {
                        "name": metric.name,
                        "value": getattr(
                            metric, "value", getattr(metric, "values", None)
                        ),
                    }
                    for metric in self.get_metrics()
                    if metric.name.startswith("vllm:spec_decode_")
                ]
                bench.launch.store_json(
                    root / "untimed-counters.json",
                    {"protocol_signature": protocol_signature(), "counters": counters},
                )
            return result

    with patch.object(vllm, "LLM", MeasuredLLM):
        graph_job(bench.evaluate, exp, root / "cudagraph.json", exp.draft_tokens)


def audit_worker(manifest):
    root = bench.root_for(AUDIT)
    bench.launch.store_json(root / "serving-runtime.json", manifest)
    bench.launch.store_json(
        root / "serving-protocol.json", {"protocol_signature": protocol_signature()}
    )
    deadline = time.monotonic() + EVAL_HOURS * 3600

    def audit_stage(python, option, env):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Draft sweep audit exceeded its time limit")
        bench.child([python, "-m", LAUNCHER, *option], env, remaining / 3600)

    audit_stage(bench.launch.PYTHON, ["--reference-worker"], os.environ.copy())
    for arm in AUDIT_ARMS:
        audit_stage(
            runtime.RUNTIME / "bin/python", ["--audit-arm", arm], serving_env(manifest)
        )
    bench.launch.store_json(
        root / "evaluated.json",
        {
            "design_signature": training.signature(),
            "runtime_signature": runtime.signature(),
            "checkpoint_sha256": bench.trained()["checkpoint_sha256"],
            "arms": list(AUDIT_ARMS),
        },
    )
    check_gate(manifest)


def serving_env(manifest):
    env = runtime.runtime_environment()
    site = manifest["site_packages"]
    env.update(
        SPS_VLLM_SOURCE=site,
        VLLM_USE_V2_MODEL_RUNNER="0",
        PYTHONPATH=os.pathsep.join([site, env.get("PYTHONPATH", "")]),
    )
    return env


def worker(phase, exp):
    launch = bench.launch
    if phase in {"probe", "train"}:
        if complete(phase, exp):
            return
        command = [
            launch.PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={exp.num_gpus}",
            "-m",
            "scripts.draft_sweep_train",
        ]
        if phase == "probe":
            command.append("--probe")
        return bench.child(
            command,
            os.environ.copy(),
            c.PROBE_HOURS if phase == "probe" else c.TRAIN_HOURS,
        )
    manifest = graph.verify_runtime()
    if complete(phase, exp, manifest):
        return
    if phase == "audit":
        return audit_worker(manifest)
    root = bench.root_for(exp)
    bench.launch.store_json(root / "serving-runtime.json", manifest)
    bench.launch.store_json(
        root / "serving-protocol.json", {"protocol_signature": protocol_signature()}
    )
    bench.child(
        [runtime.RUNTIME / "bin/python", "-m", LAUNCHER, "--eval-worker", exp.name],
        serving_env(manifest),
        EVAL_HOURS,
    )


def complete(phase, exp, manifest=None):
    if phase in {"probe", "train"}:
        marker = (
            ROOT / "prepared.json"
            if phase == "probe"
            else bench.root_for(exp) / "trained.json"
        )
        if not marker.exists():
            return False
        if json.loads(marker.read_text())["design_signature"] != training.signature():
            raise RuntimeError(f"Stale marker: {marker}")
        return True
    if not bench.completed(exp, manifest):
        return False
    if phase == "eval":
        graph.validate_evidence(bench.root_for(exp) / "cudagraph.json")
        if exp.draft_tokens:
            untimed_counters(exp)
    return True


def candidates(phase):
    if phase in {"probe", "train"}:
        return [TRAINER]
    return [AUDIT] if phase == "audit" else experiment_configs


def payload(phase, exp, staged):
    launch = bench.launch
    body = (
        run.payload(phase, exp, staged, "d.tarasov", "mrsndmn")
        if phase in {"probe", "train"}
        else bench.payload(exp, staged, phase)
    )
    identity = (
        training.signature() if phase in {"probe", "train"} else protocol_signature()
    )[:16]
    body["job_desc"] = (
        f"sps-draft-sweep {phase} {exp.name} {identity} #d.tarasov @mrsndmn"
    )
    body["script"] = f"cd {shlex.quote(str(staged))} && " + shlex.join(
        [
            str(launch.PYTHON),
            "-m",
            LAUNCHER,
            "--worker",
            "--phase",
            phase,
            "--filter",
            exp.name,
        ]
    )
    return body


def submit(args, choices):
    launch = bench.launch
    phase = args.phase
    commit = launch.git("rev-parse", "HEAD")
    staged = launch.STAGING_ROOT / commit
    if args.dry:
        for exp in choices:
            print(json.dumps(payload(phase, exp, staged), indent=2))
        return
    manifest = None
    if phase == "train":
        if not complete("probe", TRAINER):
            raise RuntimeError("Finish the 9-position probe first")
        probe = json.loads((ROOT / "prepared.json").read_text())
        seconds = probe["embedding"] * c.STAGE_STEPS + probe["full"] * (
            c.TRAIN_STEPS - c.STAGE_STEPS
        )
        if (
            min(probe["embedding"], probe["full"]) <= 0
            or 2 * seconds > c.TRAIN_HOURS * 3600
        ):
            raise RuntimeError("Probe does not fit the reserved training time")
    if phase in {"audit", "eval"}:
        bench.trained()
        manifest = graph.verify_runtime()
        if phase == "eval":
            check_gate(manifest)
    pending = [exp for exp in choices if not complete(phase, exp, manifest)]
    if not pending:
        print("__MLS_JOBS_JSON__")
        print(json.dumps({"phase": phase, "jobs": [], "deferred": []}), flush=True)
        return

    from mls.manager.job.utils import training_job_api_from_profile

    client, options = training_job_api_from_profile("default")
    launch.disable_submission_retries(client)
    active = {}
    for status in ("Pending", "Running"):
        response = client.get_list_jobs(options["region"], None, status, 1000, 0)
        if "jobs" not in response:
            raise RuntimeError(f"Cannot establish {status} jobs")
        active.update(
            {
                launch.normalize(row["job_desc"]): row["job_name"]
                for row in response["jobs"]
            }
        )
    launch.reconcile_receipts(client, active)
    launch.reconcile_intents(active)
    staged = launch.stage_repo(Path(launch.git("rev-parse", "--show-toplevel")), commit)
    occupied = sum(key.startswith("sps-draft-sweep eval ") for key in active)
    jobs = []
    deferred = []
    for exp in pending:
        body = payload(phase, exp, staged)
        identity = launch.normalize(body["job_desc"])
        arm_prefix = identity.rsplit(" ", 1)[0] + " "
        if any(key.startswith(arm_prefix) and key != identity for key in active):
            raise RuntimeError(f"An older design is already running for {exp.name}")
        if identity in active:
            jobs.append(
                {"name": exp.name, "job_name": active[identity], "reused": True}
            )
            continue
        if complete(phase, exp, manifest):
            continue
        if phase == "eval" and occupied >= bench.MAX_ACTIVE_JOBS:
            deferred.append(exp.name)
            continue
        body["region"] = options["region"]
        run.reserve("eval" if phase == "audit" else phase, exp)
        receipt_phase = (
            "prepare" if phase == "probe" else "eval" if phase == "audit" else phase
        )
        result = launch.submit_payload(client, body, receipt_phase, exp.name, commit)
        occupied += 1
        jobs.append(result)
        print(json.dumps(result), flush=True)
    print("__MLS_JOBS_JSON__")
    print(json.dumps({"phase": phase, "jobs": jobs, "deferred": deferred}), flush=True)


def untimed_counters(exp):
    path = bench.root_for(exp) / "untimed-counters.json"
    data = json.loads(path.read_text())
    if data["protocol_signature"] != protocol_signature():
        raise RuntimeError(
            f"Untimed counter snapshot uses another protocol: {exp.name}"
        )
    return data["counters"]


def counter_metrics(result, draft, untimed=()):
    if not draft:
        return {"tau": None, "acceptance_rate": None, "acceptance_by_position": None}

    def read(records):
        return {
            row["name"].removeprefix("vllm:spec_decode_"): row["value"]
            for row in records
        }

    totals = read(result["acceptance_counters_including_warmup"])
    warmup = read(untimed)
    rounds, proposed, accepted = (
        totals[key] - warmup[key]
        for key in ("num_drafts", "num_draft_tokens", "num_accepted_tokens")
    )
    positions = [
        total - warm
        for total, warm in zip(
            totals["num_accepted_tokens_per_pos"],
            warmup["num_accepted_tokens_per_pos"],
            strict=True,
        )
    ]
    if not (
        rounds > 0
        and 0 < proposed <= rounds * draft
        and len(positions) == draft
        and sum(positions) == accepted
        and 0 <= accepted <= proposed
        and all(0 <= value <= rounds for value in positions)
        and list(positions) == sorted(positions, reverse=True)
    ):
        raise ValueError("Inconsistent speculative counters")
    return {
        "tau": 1 + accepted / rounds,
        "acceptance_rate": accepted / proposed,
        "acceptance_by_position": [value / rounds for value in positions],
    }


def report():
    manifest = graph.verify_runtime()
    results = {}
    export = bench.trained()["export"]
    identity = None
    for exp in experiment_configs:
        if not complete("eval", exp, manifest):
            raise RuntimeError(f"Incomplete arm: {exp.name}")
        root = bench.root_for(exp)
        measured = json.loads((root / "manifest.json").read_text())
        args = measured["arguments"]
        expected_model = (
            str(export_for(exp.arm, exp.draft_tokens, export))
            if exp.arm.startswith("sps")
            else c.MODEL
        )
        if args["model"] != expected_model:
            raise ValueError(f"Wrong verifier/export for {exp.name}")
        if (args.get("speculative_config") or {}).get(
            "num_speculative_tokens", 0
        ) != exp.draft_tokens:
            raise ValueError(f"Wrong draft length for {exp.name}")
        trial_identity = tuple(
            measured[key]
            for key in (
                "indices",
                "dataset_revision",
                "checkpoint_sha256",
                "gpu",
                "vllm_source",
                "torch_version",
            )
        )
        if identity is not None and identity != trial_identity:
            raise ValueError(f"Different workload or runtime for {exp.name}")
        identity = trial_identity
        results[exp.name] = json.loads((root / "evaluated.json").read_text())

    rows = []
    for exp in experiment_configs:
        result = results[exp.name]
        trials = result["trials"]
        baseline = results[
            f"{'sps' if exp.arm.startswith('sps') else 'ar'}-{exp.mode}"
        ]["trials"]
        if len(trials) != c.REPEATS or any(
            row["repeat"] != i or row["count"] != c.EVAL_COUNT
            for i, row in enumerate(trials)
        ):
            raise ValueError(f"Incomplete trials for {exp.name}")
        rows.append(
            dict(
                arm=exp.arm,
                mode=exp.mode,
                draft_tokens=exp.draft_tokens,
                decode_tokens_per_second=median(
                    row["decode_tokens_per_second"] for row in trials
                ),
                speedup=median(
                    row["decode_tokens_per_second"] / base["decode_tokens_per_second"]
                    for row, base in zip(trials, baseline, strict=True)
                ),
                accuracy=trials[0]["accuracy"],
                **counter_metrics(
                    result,
                    exp.draft_tokens,
                    untimed_counters(exp) if exp.draft_tokens else (),
                ),
            )
        )
    target = ROOT / "reports"
    bench.launch.store_json(
        target / "results.json",
        {
            "design_signature": training.signature(),
            "protocol_signature": protocol_signature(),
            "rows": rows,
            "acceptance_scope": "Timed requests only. Per-position denominator is all timed drafts.",
        },
    )
    with (target / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, default="probe")
    parser.add_argument("--filter")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry", action="store_true")
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--worker", action="store_true")
    mode.add_argument("--reference-worker", action="store_true")
    mode.add_argument("--audit-arm", choices=AUDIT_ARMS)
    mode.add_argument("--eval-worker", choices=[exp.name for exp in experiment_configs])
    args = parser.parse_args()
    configure()
    if args.report:
        return report()
    if args.reference_worker:
        prepare_short_export(bench.trained()["export"])
        return bench.make_reference()
    if args.audit_arm:
        arm = args.audit_arm
        draft = AUDIT_ARMS[arm]
        return graph_job(
            bench.audit_arm, arm, bench.root_for(AUDIT) / f"graphs-{arm}.json", draft
        )
    if args.eval_worker:
        exp = next(exp for exp in experiment_configs if exp.name == args.eval_worker)
        return evaluate(exp)

    choices = [
        exp
        for exp in candidates(args.phase)
        if args.filter is None or args.filter == exp.name
    ]
    if not choices or (args.worker and args.filter is None):
        parser.error("Select a configured experiment by its exact name")
    if args.worker:
        return worker(args.phase, choices[0])
    if args.dry:
        return submit(args, choices)
    bench.launch.ARTIFACTS_ROOT = ROOT
    bench.launch.RUN_ROOT = ROOT / "runs"
    with bench.launch.submission_lock():
        return submit(args, choices)


if __name__ == "__main__":
    main()
