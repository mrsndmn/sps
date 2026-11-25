"""One-GPU final-checkpoint IFEval and held-out NLL for A and B."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import time

from scripts import ifeval_config as ic
from scripts import instruct_config as source
from scripts import mlspace_launch as launch
from scripts import real_readout_run as train
from scripts import vllm_sps_common as common
from scripts import vllm_sps_config as c

ROOT = train.ROOT / "endpoint-eval"
experiment_configs = [c.Experiment(f"{arm}-{mode}", arm, mode)
                      for arm in ("A", "B") for mode in ic.MODES]


def endpoint(exp):
    folder = train.OUT if exp.arm == "A" else c.ROOT / "runs/out/qwen3_sps_mtp"
    marker = json.loads((folder / "trained.json").read_text())
    expected = train.signature() if exp.arm == "A" else common.signature()
    if (marker["design_signature"] != expected or marker["checkpoint"] != str(folder / "final.pt")
            or marker.get("step", marker.get("steps")) != c.TRAIN_STEPS):
        raise ValueError("Endpoint provenance mismatch")
    return marker


def identity(exp, marker=None):
    if marker is None:
        marker = endpoint(exp)
    sources = ("real_readout_eval", "instruct_evaluate", "instruct_generation", "mmlu_generation",
               "provenance_evaluate", "ifeval_config")
    return dict(checkpoint=marker["checkpoint_sha256"], training=marker["design_signature"],
                arm=exp.arm, mode=exp.mode, generation=source.GENERATION[exp.mode],
                files={s: common.file_hash(Path(__file__).with_name(s + ".py")) for s in sources})


def load_endpoint(exp, marker):
    import torch
    from scripts.instruct_common import load_model
    from scripts.real_readout_model import RealReadoutMTP
    from scripts.vllm_sps_model import QwenSPSMTP

    path = marker["checkpoint"]
    if common.file_hash(path) != marker["checkpoint_sha256"]:
        raise ValueError("Endpoint checksum mismatch")
    state = torch.load(path, mmap=True, map_location="cpu", weights_only=False)
    if state["step"] != c.TRAIN_STEPS or state["design_signature"] != marker["design_signature"]:
        raise ValueError("Checkpoint step or signature mismatch")
    native = load_model("continued_ar" if exp.arm == "A" else "staged_sps", "cpu")
    packed = (RealReadoutMTP(native.backbone, c.HORIZONS) if exp.arm == "A"
              else QwenSPSMTP(native, c.HORIZONS))
    packed.load_state_dict(state["model"], strict=True)
    packed.set_stage(False)
    return native.to("cuda").eval().requires_grad_(False), packed.to("cuda").eval().requires_grad_(False)


def horizon_losses(model, rows, device):
    import torch
    from scripts.vllm_sps_model import horizon_targets

    sums = [0.] * c.HORIZONS
    counts = [0] * c.HORIZONS
    for row in rows:
        tokens = torch.tensor([row["input_ids"]], device=device)
        assistant = torch.tensor([row["loss_mask"]], device=device)
        labels = horizon_targets(tokens, assistant, c.HORIZONS)
        for h in range(c.HORIZONS):
            selected = torch.full_like(labels, -100)
            selected[:, :, h] = labels[:, :, h]
            if not (selected != -100).any():
                continue
            value, count = model(tokens[:, :-1], targets=selected)
            sums[h] += value.item()
            counts[h] += count
    return [dict(horizon=h+1, tokens=n, nll=value/n) for h, (n, value) in enumerate(zip(counts, sums))]


def evaluate(exp):
    import torch
    from datasets import load_dataset
    from langdetect import DetectorFactory
    from transformers import AutoTokenizer
    from scripts.instruct_common import nll
    from scripts.instruct_evaluate import ensure_ifeval_resources, evaluate_generation
    from scripts.provenance_evaluate import protocol

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible GPU required")
    root = ROOT / "runs/out" / exp.name
    marker = endpoint(exp)
    ident = identity(exp, marker)
    protocol()
    meta = root / "manifest.json"
    if meta.exists() and json.loads(meta.read_text()) != ident:
        raise ValueError("Incompatible saved evaluation")
    launch.store_json(meta, ident)
    native, packed = load_endpoint(exp, marker)
    DetectorFactory.seed = source.SEED
    ensure_ifeval_resources()
    tokenizer = AutoTokenizer.from_pretrained(c.MODEL, revision=c.REVISION)
    docs = list(load_dataset(source.IFEVAL_DATASET, revision=source.IFEVAL_REVISION, split="train"))
    sig = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if exp.mode == "nonthinking":
            rows = common.data("test")
            sequential = nll(native, rows, torch.device("cuda"))
            horizons = horizon_losses(packed, rows, torch.device("cuda"))
            if horizons[0]["tokens"] != sequential["tokens"]:
                raise ValueError("First-horizon target count mismatch")
            launch.store_json(root / "heldout.json", dict(sequential=sequential, horizons=horizons))
        metrics = evaluate_generation(native, tokenizer, docs, "ifeval", exp.mode,
                                      source.GENERATION[exp.mode], root, sig, c.SEED,
                                      time.monotonic(), ic.EVAL_HOURS)
    launch.store_json(root / "evaluated.json", dict(identity=ident, metrics=metrics,
                                                   artifacts=artifact_hashes(root, exp)))


def artifact_hashes(root, exp):
    names = [f"ifeval-{exp.mode}.jsonl", f"ifeval-{exp.mode}.json"]
    if exp.mode == "nonthinking":
        names.append("heldout.json")
    return {name: common.file_hash(root / name) for name in names}


def submit(experiments):
    from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

    client, options = training_job_api_from_profile("default")
    launch.disable_submission_retries(client)
    active = {launch.normalize(j["job_desc"]): j["job_name"]
              for j in get_in_progress_jobs(client_profile="default")}
    launch.reconcile_receipts(client, active)
    launch.reconcile_intents(active)
    commit = launch.git("rev-parse", "HEAD")
    staged = launch.stage_repo(Path(launch.git("rev-parse", "--show-toplevel")), commit)
    for exp in experiments:
        ident = identity(exp)
        marker = ROOT / "runs/out" / exp.name / "evaluated.json"
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved["identity"] != ident or saved["artifacts"] != artifact_hashes(marker.parent, exp):
                raise ValueError("Stale evaluation marker")
            continue
        body = launch.payload("eval", exp.name, staged, "d.tarasov", "mrsndmn")
        body.update(instance_type=launch.INSTANCE_TYPES_BY_NUM_GPUS[1], region=options["region"],
                    job_desc=f"sps-real-readout-eval {exp.name} #d.tarasov @mrsndmn",
                    script=f"cd {shlex.quote(str(staged))} && " + shlex.join([
                        "timeout", f"{ic.EVAL_HOURS}h", str(launch.PYTHON), "-m",
                        "scripts.real_readout_eval", "--worker", "--filter", exp.name]))
        body["env_variables"]["SPS_EXPERIMENT_SUITE"] = ic.SOURCE_SUITE
        key = launch.normalize(body["job_desc"])
        if key in active:
            print(exp.name, active[key], flush=True)
        else:
            print(json.dumps(launch.submit_payload(client, body, "eval", exp.name, commit)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter", choices=[e.name for e in experiment_configs])
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    selected = [e for e in experiment_configs if args.filter in (None, e.name)]
    if args.worker:
        if len(selected) != 1:
            parser.error("Worker requires one experiment")
        return evaluate(selected[0])
    launch.ARTIFACTS_ROOT, launch.RUN_ROOT = ROOT, ROOT / "runs"
    with launch.submission_lock():
        submit(selected)


if __name__ == "__main__":
    main()
