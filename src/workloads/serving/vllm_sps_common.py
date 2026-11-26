"""Pinned parent/data inputs and isolated SPS-MTP artifact identities."""

import hashlib
import json
import math
from pathlib import Path

from experiments import vllm_sps_config as c

_LEGACY_SCHEDULE = {"training_steps": 512, "stage_steps": 102, "global_batch": 16}
_LEGACY_PROBE = {
    "global_batch": _LEGACY_SCHEDULE["global_batch"],
    "num_gpus": 4,
    "context": 2048,
    "backend": "flex_attention",
}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def training_schedule():
    return {"training_steps": c.family_config.TRAIN_STEPS, "stage_steps": c.family_config.STAGE_STEPS, "global_batch": c.family_config.GLOBAL_BATCH}


def recorded_schedule(record):
    return {name: record.get(name, default) for name, default in _LEGACY_SCHEDULE.items()}


def trained():
    marker = c.family_config.ROOT / "runs/out/qwen3_sps_mtp/trained.json"
    record = json.loads(marker.read_text())
    checkpoint = Path(record["checkpoint"])
    if (
        record["steps"] != c.family_config.TRAIN_STEPS
        or recorded_schedule(record) != training_schedule()
        or record["horizons"] != c.family_config.HORIZONS
        or record["parent"] != str(c.family_config.PARENT)
        or checkpoint.name != "final.pt"
        or Path(record["export"]) != checkpoint.with_name("hf")
        or not record["checkpoint_sha256"]
    ):
        raise RuntimeError(f"Training checkpoint does not match the configured run: {marker}")
    return record


def checked_checkpoint(record):
    checkpoint = Path(record["checkpoint"])
    if file_hash(checkpoint) != record["checkpoint_sha256"]:
        raise RuntimeError(f"Trained checkpoint changed: {checkpoint}")
    return checkpoint


def matches_training_input(state):
    return (
        state["source"] == c.family_config.MODEL
        and state["revision"] == c.family_config.REVISION
        and state["parent"] == str(c.family_config.PARENT)
        and state["horizons"] == c.family_config.HORIZONS
    )


def probe_workload(num_gpus):
    return {
        "horizons": c.family_config.HORIZONS,
        "global_batch": c.family_config.GLOBAL_BATCH,
        "num_gpus": num_gpus,
        "context": c.source.family_config.CONTEXT,
        "backend": c.family_config.BACKEND,
    }


def probe_seconds(path, *, num_gpus=None, legacy_horizons=4):
    probe = json.loads(path.read_text())
    expected = probe_workload(c.experiment_configs[0].num_gpus if num_gpus is None else num_gpus)
    legacy = dict(_LEGACY_PROBE, horizons=legacy_horizons)
    recorded = {field: probe.get(field, default) for field, default in legacy.items()}
    if recorded != expected:
        raise RuntimeError(f"Training probe uses another workload: {path}")
    if "seconds" in probe:
        embedding, full = probe["seconds"]
    else:
        embedding, full = probe["embedding"], probe["full"]
    if any(not math.isfinite(value) or value <= 0 for value in (embedding, full)):
        raise RuntimeError(f"Invalid training probe timings: {path}")
    return embedding * c.family_config.STAGE_STEPS + full * (c.family_config.TRAIN_STEPS - c.family_config.STAGE_STEPS)


def recorded_evaluation_protocol(record):
    field = "evaluation_protocol" if "evaluation_protocol" in record else "design_signature"
    digest = record.get(field)
    if not isinstance(digest, str) or not digest:
        raise RuntimeError(f"Evaluation is missing {field}")
    return field, digest


def evaluation_record(exp, checkpoint_sha256):
    root = c.family_config.ROOT / "runs/out" / exp.name
    record = json.loads((root / "evaluated.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if record["experiment"] != vars(exp) or manifest["experiment"] != vars(exp):
        raise RuntimeError(f"Evaluation used another experiment: {root}")
    protocol = recorded_evaluation_protocol(manifest)
    if recorded_evaluation_protocol(record) != protocol:
        raise RuntimeError(f"Evaluation results and manifest use different protocols: {root}")
    recorded_checksum = manifest["checkpoint_sha256"]
    legacy_checksum = recorded_checksum if protocol[0] == "design_signature" else None
    result_checksum = record.get("checkpoint_sha256", legacy_checksum)
    if recorded_checksum != checkpoint_sha256 or result_checksum != checkpoint_sha256:
        raise RuntimeError(f"Evaluation used another checkpoint: {root}")
    return record, manifest


def data(split):
    root = c.family_config.SOURCE_ROOT / "runs/data"
    manifest = json.loads((root / "manifest.json").read_text())
    path = root / f"{split}.jsonl"
    if (
        manifest['model'] != c.family_config.MODEL or manifest['revision'] != c.family_config.REVISION or file_hash(path) != manifest['sha256'][split]
    ):
        raise RuntimeError("Instruction data provenance mismatch")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) != manifest["rows"][split]:
        raise RuntimeError("Instruction data count mismatch")
    return records


def load_model(device, backend=c.family_config.BACKEND):
    import torch

    from workloads.instruction.instruct_common import load_model as original
    from workloads.serving.vllm_sps_model import QwenSPSMTP

    state = torch.load(c.family_config.PARENT, map_location="cpu", mmap=True, weights_only=False)
    if (
        state['source'] != c.family_config.MODEL or state['revision'] != c.family_config.REVISION or state['step'] != c.family_config.PARENT_STEP
    ):
        raise RuntimeError("SPS parent provenance mismatch")
    model = original("staged_sps", "cpu")
    model.load_state_dict(state["model"])
    del state
    return QwenSPSMTP(model, c.family_config.HORIZONS, backend=backend).to(device)


def export(model, destination):
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    destination.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(destination, safe_serialization=True)
    embeddings = (
        __import__("torch")
        .cat((model.prediction[None], model.new_embeddings))
        .detach()
        .cpu()
        .contiguous()
    )
    save_file(
        {"sps_prediction_embeddings": embeddings}, destination / "sps.safetensors"
    )
    path = destination / "config.json"
    config = json.loads(path.read_text())
    config.update(
        architectures=["Qwen3SPSForCausalLM"],
        sps_horizons=model.horizons,
    )
    path.write_text(json.dumps(config, indent=2))
    # A sharded HF checkpoint's index must include the additional parameter.
    index = destination / "model.safetensors.index.json"
    if index.exists():
        value = json.loads(index.read_text())
        value["weight_map"]["sps_prediction_embeddings"] = "sps.safetensors"
        value["metadata"]["total_size"] += (
            embeddings.numel() * embeddings.element_size()
        )
        index.write_text(json.dumps(value, indent=2))
    AutoTokenizer.from_pretrained(c.family_config.MODEL, revision=c.family_config.REVISION).save_pretrained(
        destination
    )
