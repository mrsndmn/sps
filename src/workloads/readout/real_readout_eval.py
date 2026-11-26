"""One-GPU final-checkpoint IFEval and held-out NLL for A and B."""
from orchestration.named.publication import work_path
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import time

from experiments import ifeval_config as ic
from experiments import instruct_config as source
from orchestration import mlspace_launch as launch
from workloads.readout import real_readout_run as train
from workloads.serving import vllm_sps_common as common
from experiments import vllm_sps_config as c
from experiments.real_readout_eval import family_config as real_readout_eval_settings
from experiments.real_readout_eval import experiment_configs
ROOT = real_readout_eval_settings.ROOT


def endpoint(exp):
    if exp.arm == "A":
        return train.trained()
    marker = common.trained()
    if Path(marker["checkpoint"]) != c.family_config.ROOT / "runs/out/qwen3_sps_mtp/final.pt":
        raise ValueError("Unexpected SPS endpoint path")
    return marker


def identity(exp, marker=None):
    if marker is None:
        marker = endpoint(exp)
    return dict(checkpoint=marker["checkpoint_sha256"],
                arm=exp.arm, mode=exp.mode, generation=source.family_config.GENERATION[exp.mode])


def load_endpoint(exp, marker):
    import torch
    from workloads.instruction.instruct_common import load_model
    from workloads.readout.real_readout_model import RealReadoutMTP
    from workloads.serving.vllm_sps_model import QwenSPSMTP

    path = common.checked_checkpoint(marker)
    state = torch.load(path, mmap=True, map_location="cpu", weights_only=False)
    if state["step"] != c.family_config.TRAIN_STEPS or state["parent"] != marker["parent"]:
        raise ValueError("Checkpoint step or parent mismatch")
    if exp.arm == "B" and not common.matches_training_input(state):
        raise ValueError("SPS checkpoint does not match the configured model")
    native = load_model("continued_ar" if exp.arm == "A" else "staged_sps", "cpu")
    packed = (RealReadoutMTP(native.backbone, c.family_config.HORIZONS) if exp.arm == "A"
              else QwenSPSMTP(native, c.family_config.HORIZONS))
    packed.load_state_dict(state["model"], strict=True)
    packed.set_stage(False)
    return native.to("cuda").eval().requires_grad_(False), packed.to("cuda").eval().requires_grad_(False)


def horizon_losses(model, rows, device):
    import torch
    from workloads.serving.vllm_sps_model import horizon_targets

    sums = [0.] * c.family_config.HORIZONS
    counts = [0] * c.family_config.HORIZONS
    for row in rows:
        tokens = torch.tensor([row["input_ids"]], device=device)
        assistant = torch.tensor([row["loss_mask"]], device=device)
        labels = horizon_targets(tokens, assistant, c.family_config.HORIZONS)
        for h in range(c.family_config.HORIZONS):
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
    from workloads.instruction.instruct_common import nll
    from workloads.instruction.instruct_evaluate import ensure_ifeval_resources, evaluate_generation
    from workloads.provenance.provenance_evaluate import protocol

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible GPU required")
    root = work_path(ROOT / "runs/out" / exp.name)
    marker = endpoint(exp)
    ident = identity(exp, marker)
    protocol()
    meta = root / "manifest.json"
    if not meta.exists() and (root / f"ifeval-{exp.mode}.jsonl").exists():
        raise ValueError("Saved IFEval answers have no checkpoint manifest")
    if meta.exists() and json.loads(meta.read_text()) != ident:
        raise ValueError("Incompatible saved evaluation")
    launch.store_json(meta, ident)
    native, packed = load_endpoint(exp, marker)
    DetectorFactory.seed = source.family_config.SEED
    ensure_ifeval_resources()
    tokenizer = AutoTokenizer.from_pretrained(c.family_config.MODEL, revision=c.family_config.REVISION)
    docs = list(load_dataset(source.family_config.IFEVAL_DATASET, revision=source.family_config.IFEVAL_REVISION, split="train"))
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
                                      source.family_config.GENERATION[exp.mode], root, sig, c.family_config.SEED,
                                      time.monotonic(), ic.family_config.EVAL_HOURS)
    launch.store_json(root / "evaluated.json", dict(identity=ident, metrics=metrics,
                                                   artifacts=artifact_hashes(root, exp)))


def artifact_hashes(root, exp):
    names = [f"ifeval-{exp.mode}.jsonl", f"ifeval-{exp.mode}.json"]
    if exp.mode == "nonthinking":
        names.append("heldout.json")
    return {name: common.file_hash(root / name) for name in names}


def submit(experiments):
    experiments = [exp for exp in experiments if launch.pending(ROOT / "runs/out" / exp.name)]
    if not experiments:
        return

    commit = launch.git("rev-parse", "HEAD")
    batch = launch.SubmissionBatch(commit)
    staged = launch.STAGING_ROOT / commit
    for exp in experiments:
        ident = identity(exp)
        marker = work_path(ROOT / "runs/out" / exp.name) / "evaluated.json"
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved["identity"] != ident or saved["artifacts"] != artifact_hashes(marker.parent, exp):
                raise ValueError("Stale evaluation marker")
            continue
        body = payload(exp, staged)
        print(json.dumps(batch.submit(body, "eval", exp.name, ROOT / "runs/out" / exp.name)), flush=True)


def payload(exp, staged):
    body = launch.payload("eval", exp.name, staged, "d.tarasov", "mrsndmn")
    body.update(instance_type=launch.INSTANCE_TYPES_BY_NUM_GPUS[1],
                job_desc=f"sps-real-readout-eval {exp.name} #d.tarasov @mrsndmn",
                script=f"cd {shlex.quote(str(staged))} && " + shlex.join([
                    "timeout", f"{ic.family_config.EVAL_HOURS}h", str(launch.PYTHON), "scripts/readout/real_readout_eval.py", "--worker", "--filter", exp.name]))
    body["env_variables"]["SPS_EXPERIMENT_SUITE"] = ic.family_config.SOURCE_SUITE
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter", choices=[e.name for e in experiment_configs])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    selected = [e for e in experiment_configs if args.filter in (None, e.name)]
    if args.dry:
        staged = launch.STAGING_ROOT / launch.git("rev-parse", "HEAD")
        for exp in selected:
            launch.print_preview(payload(exp, staged), ROOT / "runs/out" / exp.name, staged, "eval")
        return
    if args.worker:
        if len(selected) != 1:
            parser.error("Worker requires one experiment")
        return evaluate(selected[0])
    launch.ARTIFACTS_ROOT, launch.RUN_ROOT = ROOT, ROOT / "runs"
    with launch.submission_lock():
        submit(selected)
