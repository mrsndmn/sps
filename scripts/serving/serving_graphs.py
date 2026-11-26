"""Isolated decode CUDA-graph gate and matched vLLM benchmark."""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts import serving_benchmark as bench
from scripts import serving_runtime as runtime
from scripts import vllm_sps_common as common
from scripts import vllm_sps_config as c

EAGER_ROOT = c.ROOT
ROOT = EAGER_ROOT.parent / "vllm-sps-qwen3-cudagraphs"
IMPLEMENTATION = "31ba03a874f775372154f0edf3e17d436f2d7462"
SITE = ROOT / "runtime/site-packages"
COMPILATION = {
    "mode": 0,
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [1, 2, 3, 4],
}
experiment_configs = bench.experiment_configs
BASE_VERIFY = bench.verify_runtime
BASE_PROTOCOL = bench.protocol_signature
BASE_CHILD = bench.child
BASE_PAYLOAD = bench.payload
BASE_COMPLETED = bench.completed
BASE_GATE = bench.check_gate
BASE_AUDIT = bench.audit_arm
BASE_EVALUATE = bench.evaluate


def protocol_signature():
    return hashlib.sha256(
        json.dumps(
            [
                BASE_PROTOCOL(),
                common.file_hash(Path(__file__)),
                IMPLEMENTATION,
                COMPILATION,
            ]
        ).encode()
    ).hexdigest()


def prepare_source():
    """Copy Python sources; share immutable native binaries with the exact wheel."""
    base = BASE_VERIFY()
    changed = bench.launch.git(
        "-C",
        str(c.VLLM_WORKTREE),
        "diff",
        "--name-only",
        c.VLLM_COMMIT,
        IMPLEMENTATION,
    ).splitlines()
    if any(
        not p.startswith(("docs/", "tests/")) and p not in runtime.OVERLAY
        for p in changed
    ):
        raise RuntimeError("Unaccounted native/runtime change in graph implementation")
    if not SITE.exists():
        SITE.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SITE.parent) as tmp:
            staged = Path(tmp) / "site-packages"

            def copy_file(source, target):
                if Path(source).suffix == ".py":
                    return shutil.copy2(source, target)
                os.symlink(source, target)
                return target

            shutil.copytree(
                Path(base["site_packages"]) / "vllm",
                staged / "vllm",
                copy_function=copy_file,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            hashes = {}
            for name in runtime.OVERLAY:
                content = subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(c.VLLM_WORKTREE),
                        "show",
                        f"{IMPLEMENTATION}:{name}",
                    ]
                )
                target = staged / name
                # All overlay files are ordinary copies, never shared inodes.
                target.write_bytes(content)
                hashes[name] = hashlib.sha256(content).hexdigest()
            bench.launch.store_json(
                staged / "graph-source.json",
                {
                    "implementation": IMPLEMENTATION,
                    "base_runtime": base,
                    "site_packages": str(SITE),
                    "overlay_sha256": hashes,
                },
            )
            staged.rename(SITE)
    verify_runtime()
    trained = bench.trained()
    target = ROOT / "runs/out/qwen3_sps_mtp/trained.json"
    if target.exists() and json.loads(target.read_text()) != trained:
        raise RuntimeError("Graph benchmark refers to another trained model")
    bench.launch.store_json(target, trained)


def verify_runtime():
    base = BASE_VERIFY()
    data = json.loads((SITE / "graph-source.json").read_text())
    if (
        data["implementation"] != IMPLEMENTATION
        or data["base_runtime"] != base
        or data["site_packages"] != str(SITE)
        or set(data["overlay_sha256"]) != set(runtime.OVERLAY)
    ):
        raise RuntimeError("Graph source provenance mismatch")
    for name, expected in data["overlay_sha256"].items():
        if common.file_hash(SITE / name) != expected:
            raise RuntimeError(f"Graph source changed: {name}")
    native = "vllm/_C_stable_libtorch.abi3.so"
    if common.file_hash(SITE / native) != common.file_hash(
        Path(base["site_packages"]) / native
    ):
        raise RuntimeError("Graph native extension differs from the exact wheel")
    return dict(data, design_signature=runtime.signature())


def graph_arguments(original, *args, **kwargs):
    result = original(*args, **kwargs)
    result.update(
        enforce_eager=False,
        compilation_config=dict(COMPILATION),
        worker_extension_cls="scripts.serving_graphs.GraphReplayWorkerExtension",
    )
    return result


def start_tracking(worker):
    """Instrument actual target graph replay only during untimed warmup."""
    import torch
    from vllm.compilation.cuda_graph import CUDAGraphWrapper

    model = worker.model_runner.model
    if not isinstance(model, CUDAGraphWrapper):
        raise TypeError("Target model has no full CUDA graph wrapper")
    entries = model.concrete_cudagraph_entries
    graphs = {id(x.cudagraph) for x in entries.values() if x.cudagraph is not None}
    if not graphs:
        raise RuntimeError("Target did not capture any CUDA graphs")
    if hasattr(worker, "_sps_graph_tracking"):
        raise RuntimeError("Graph replay tracking already active")
    original = torch.cuda.CUDAGraph.replay
    state = {
        "captured": len(graphs),
        "target_replays": 0,
        "mode": str(worker.model_runner.compilation_config.cudagraph_mode),
    }

    def replay(graph):
        if id(graph) in graphs:
            state["target_replays"] += 1
        return original(graph)

    worker._sps_graph_tracking = (original, state)
    torch.cuda.CUDAGraph.replay = replay


def stop_tracking(worker):
    import torch

    original, state = worker._sps_graph_tracking
    torch.cuda.CUDAGraph.replay = original
    del worker._sps_graph_tracking
    return state


class GraphReplayWorkerExtension:
    """Expose replay tracking through serializable named worker RPC methods."""

    def sps_start_graph_tracking(self):
        start_tracking(self)

    def sps_stop_graph_tracking(self):
        return stop_tracking(self)


def validate_evidence(path):
    data = json.loads(path.read_text())
    if data.get("protocol_signature") != protocol_signature():
        raise RuntimeError("Graph replay evidence uses another protocol")
    workers = data.get("workers", [])
    if (
        len(workers) != 1
        or workers[0].get("captured", 0) < 1
        or workers[0].get("target_replays", 0) < 1
        or "FULL" not in workers[0].get("mode", "")
    ):
        raise RuntimeError("Missing evidence of actual target CUDA graph replay")


def with_graph_engine(function, argument, evidence):
    import vllm

    from scripts import vllm_sps_evaluate as evaluator

    original = vllm.LLM

    class VerifiedGraphLLM(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.collective_rpc("sps_start_graph_tracking")
            self.graph_warmups = 0

        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            self.graph_warmups += 1
            if self.graph_warmups == c.WARMUP_REQUESTS:
                workers = self.collective_rpc("sps_stop_graph_tracking")
                bench.launch.store_json(
                    evidence,
                    {
                        "protocol_signature": protocol_signature(),
                        "workers": workers,
                    },
                )
                validate_evidence(evidence)
            return result

    with (
        patch.object(vllm, "LLM", VerifiedGraphLLM),
        patch.object(
            evaluator,
            "engine_arguments",
            side_effect=lambda *a, **k: graph_arguments(engine_arguments, *a, **k),
        ),
    ):
        function(argument)
    validate_evidence(evidence)


def audit_arm(arm):
    with_graph_engine(
        BASE_AUDIT, arm, ROOT / "runs/out" / bench.AUDIT_NAME / f"graphs-{arm}.json"
    )


def evaluate(exp):
    with_graph_engine(BASE_EVALUATE, exp, bench.root_for(exp) / "cudagraph.json")


def check_gate(manifest):
    BASE_GATE(manifest)
    for arm in ("sps", "sps_mtp"):
        validate_evidence(ROOT / "runs/out" / bench.AUDIT_NAME / f"graphs-{arm}.json")


def completed(exp, manifest):
    result = BASE_COMPLETED(exp, manifest)
    if result and exp.arm != "audit":
        validate_evidence(bench.root_for(exp) / "cudagraph.json")
    return result


def child(command, env, hours):
    rewritten = [
        "scripts.serving_graphs" if str(x) == "scripts.serving_benchmark" else x
        for x in command
    ]
    env = dict(env)
    env["PYTHONPATH"] = os.pathsep.join([str(SITE), env.get("PYTHONPATH", "")])
    return BASE_CHILD(rewritten, env, hours)


def payload(exp, staged, phase):
    body = BASE_PAYLOAD(exp, staged, phase)
    body["job_desc"] = body["job_desc"].replace(
        "sps-serving-benchmark ", "sps-serving-benchmark graphs "
    )
    body["script"] = f"cd {shlex.quote(str(staged))} && " + shlex.join(
        [
            str(bench.launch.PYTHON),
            "-m",
            "scripts.serving_graphs",
            "--worker",
            "--phase",
            phase,
            "--filter",
            exp.name,
        ]
    )
    return body


def configure():
    c.ROOT = ROOT
    bench.protocol_signature = protocol_signature
    bench.verify_runtime = verify_runtime
    bench.audit_arm = audit_arm
    bench.evaluate = evaluate
    bench.check_gate = check_gate
    bench.completed = completed
    bench.child = child
    bench.payload = payload


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prepare-source", action="store_true")
    parser.add_argument("--report", action="store_true")
    args, _ = parser.parse_known_args()
    if args.prepare_source:
        return prepare_source()
    configure()
    if args.report:
        from scripts import vllm_sps_report

        manifest = verify_runtime()
        for exp in bench.selected("eval"):
            if not completed(exp, manifest):
                raise RuntimeError(f"Incomplete graph arm: {exp.name}")
        return vllm_sps_report.report()
    bench.main()


from scripts.vllm_sps_evaluate import engine_arguments

if __name__ == "__main__":
    main()
