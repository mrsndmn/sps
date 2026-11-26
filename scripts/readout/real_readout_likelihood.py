"""Compare final native/SPS readouts with their parents on pinned zero-shot tasks."""
import argparse
import importlib.metadata
import json
import re
from pathlib import Path
import shlex
import tempfile

from scripts import instruct_config as source
from scripts import mlspace_launch as launch
from scripts import real_readout_eval as quality
from scripts import real_readout_run as train
from scripts import sched_config
from scripts import vllm_sps_common as common
from scripts import vllm_sps_config as c

INITIAL_WORKER = "73a63d156648bcd65801378ee670a9bb70d68cf3baba32b9336be0f340841baa"
ROOT = train.ROOT / "likelihood-eval"
experiment_configs = [c.Experiment(f"likelihood-{arm}", arm) for arm in ("A", "B")]


def inputs(exp, marker):
    parent = Path(marker["parent"])
    expected = train.PARENT if exp.arm == "A" else c.PARENT
    folder = parent.parents[1] / (parent.parent.name + "--quality") / "eval"
    manifest = json.loads((folder / "manifest.json").read_text())
    if (parent != expected or manifest["checkpoint"]["path"] != str(parent)
            or manifest["design_signature"] != c.PARENT_SIGNATURE or manifest["limit"] is not None
            or manifest["checkpoint"]["step"] != c.PARENT_STEP):
        raise ValueError("Parent benchmark checkpoint mismatch")
    stat = parent.stat()
    if (manifest["checkpoint"]["size"] != stat.st_size
            or manifest["checkpoint"]["mtime_ns"] != stat.st_mtime_ns):
        raise ValueError("Parent checkpoint changed since its evaluation")
    identity = dict(arm=exp.arm, endpoint=marker["checkpoint_sha256"], parent=str(parent),
                    model_sources=quality.model_sources(exp),
                    parents={t: common.file_hash(folder / f"{t}.json") for t in source.LIKELIHOOD_TASKS},
                    files={p: common.file_hash(Path(__file__).with_name(p + ".py"))
                           for p in ("real_readout_eval", "real_readout_likelihood", "instruct_evaluate")},
                    lm_eval=importlib.metadata.version("lm_eval"))
    return folder, identity


def compare(task, current, parent):
    def keys(result):
        return [(s["doc_id"], s["doc_hash"], s["prompt_hash"], s["target_hash"])
                for s in result["samples"][task]]
    ids = keys(parent)
    if not ids or len(set(ids)) != len(ids) or keys(current) != ids:
        raise ValueError(f"Different benchmark examples or prompts for {task}")
    if any(a["arguments"] != b["arguments"]
           for a, b in zip(current["samples"][task], parent["samples"][task])):
        raise ValueError(f"Benchmark choices or requests changed for {task}")
    for field in ("lm_eval_version", "transformers_version", "max_length", "tokenizer_pad_token",
                  "tokenizer_bos_token", "tokenizer_eos_token", "eot_token_id", "config"):
        if current[field] != parent[field]:
            raise ValueError(f"Different {field} for {task}")
    def protocol(result):
        return re.sub(r"(<function [^<>]+) at 0x[0-9a-f]+>", r"\1>",
                      json.dumps(result["configs"][task], sort_keys=True))
    if protocol(current) != protocol(parent):
        raise ValueError(f"Different task configuration for {task}")
    for name in ("n-shot", "versions", "higher_is_better"):
        if current[name][task] != parent[name][task]:
            raise ValueError(f"Scoring protocol changed for {task}")
    scores = {}
    for metric, baseline in parent["results"][task].items():
        if metric.endswith(",none") and "_stderr," not in metric:
            final = current["results"][task][metric]
            scores[metric] = dict(parent=baseline, final=final, delta=final-baseline)
    return dict(count=len(ids), scores=scores)


def evaluate(exp):
    import torch
    from transformers import AutoTokenizer
    from scripts.instruct_evaluate import likelihood
    if torch.cuda.device_count() != 1:
        raise RuntimeError("One visible GPU required")
    marker = quality.endpoint(exp)
    parent, ident = inputs(exp, marker)
    verify_parent(marker)
    native, packed = quality.load_endpoint(exp, marker)
    del packed
    tokenizer = AutoTokenizer.from_pretrained(c.MODEL, revision=c.REVISION)
    root = ROOT / "runs/out" / exp.name
    root.mkdir(parents=True, exist_ok=True)
    # Publish only complete validated suites. A retry never mixes task outputs.
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        folder = Path(tmp)
        likelihood(native, tokenizer, folder, limit=None)
        results = {}
        hashes = {}
        for task in source.LIKELIHOOD_TASKS:
            path = folder / f"{task}.json"
            baseline = parent / path.name
            if common.file_hash(baseline) != ident["parents"][task]:
                raise ValueError("Parent results changed during evaluation")
            results[task] = compare(task, json.loads(path.read_text()), json.loads(baseline.read_text()))
            hashes[path.name] = common.file_hash(path)
        for name in hashes:
            (folder / name).replace(root / name)
    launch.store_json(root / "evaluated.json", dict(identity=ident, results=results, artifacts=hashes))


def verify_parent(marker):
    if "parent_sha256" in marker and common.file_hash(marker["parent"]) != marker["parent_sha256"]:
        raise ValueError("Historical parent checksum mismatch")


def verify_saved(exp):
    # Independently re-score the comparison from the GPU jobs' immutable task logs.
    marker = quality.endpoint(exp)
    verify_parent(marker)
    parent, ident = inputs(exp, marker)
    root = ROOT / "runs/out" / exp.name
    saved = json.loads((root / "evaluated.json").read_text())
    actual = saved["identity"]
    old = dict(ident, files=dict(ident["files"], real_readout_likelihood=INITIAL_WORKER))
    if actual not in (ident, old):
        raise ValueError("Unrecognized likelihood output identity")
    verified = {}
    for task in source.LIKELIHOOD_TASKS:
        path = root / f"{task}.json"
        if saved["artifacts"][path.name] != common.file_hash(path):
            raise ValueError("Changed task artifact")
        verified[task] = compare(task, json.loads(path.read_text()),
                                 json.loads((parent / path.name).read_text()))
    if verified != saved["results"]:
        raise ValueError("Saved metrics differ from validated task outputs")
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B"))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    selected = [e for e in experiment_configs if args.arm in (None, e.arm)]
    if args.verify:
        for exp in selected:
            print(json.dumps({exp.name: verify_saved(exp)}))
        return
    if args.worker:
        exp, = selected
        return evaluate(exp)
    from mls.manager.job.utils import training_job_api_from_profile, get_in_progress_jobs
    launch.ARTIFACTS_ROOT, launch.RUN_ROOT = ROOT, ROOT / "runs"
    with launch.submission_lock():
        client, options = training_job_api_from_profile("default")
        launch.disable_submission_retries(client)
        active = {launch.normalize(j["job_desc"]): j["job_name"]
                  for j in get_in_progress_jobs(client_profile="default")}
        launch.reconcile_receipts(client, active)
        launch.reconcile_intents(active)
        commit = launch.git("rev-parse", "HEAD")
        staged = launch.stage_repo(Path(launch.git("rev-parse", "--show-toplevel")), commit)
        for exp in selected:
            _, ident = inputs(exp, quality.endpoint(exp))
            root = ROOT / "runs/out" / exp.name
            marker = root / "evaluated.json"
            if marker.exists():
                verify_saved(exp)
                continue
            body = launch.payload("eval", exp.name, staged, "d.tarasov", "mrsndmn")
            body.update(instance_type=launch.INSTANCE_TYPES_BY_NUM_GPUS[1], region=options["region"],
                        job_desc=f"sps-real-readout {exp.name} #d.tarasov @mrsndmn",
                        script=f"cd {shlex.quote(str(staged))} && " + shlex.join([
                            "timeout", f"{sched_config.WORKLOAD_HOURS['quality']}h", str(launch.PYTHON),
                            "-m", "scripts.real_readout_likelihood", "--worker", "--arm", exp.arm]))
            body["env_variables"]["SPS_EXPERIMENT_SUITE"] = "ar-sps-qwen3-1.7b-instruct"
            key = launch.normalize(body["job_desc"])
            if key in active:
                print(exp.name, active[key], flush=True)
            else:
                print(json.dumps(launch.submit_payload(client, body, "eval", exp.name, commit)), flush=True)


if __name__ == "__main__":
    main()
