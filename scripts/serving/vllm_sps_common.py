"""Pinned parent/data inputs and isolated SPS-MTP artifact identities."""

import hashlib
import json
import math
from pathlib import Path

from scripts import vllm_sps_config as c


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def trained():
    marker = c.ROOT / "runs/out/qwen3_sps_mtp/trained.json"
    record = json.loads(marker.read_text())
    checkpoint = Path(record["checkpoint"])
    if (
        record["steps"] != c.TRAIN_STEPS
        or record["horizons"] != c.HORIZONS
        or record["parent"] != str(c.PARENT)
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
        state["source"] == c.MODEL
        and state["revision"] == c.REVISION
        and state["parent"] == str(c.PARENT)
        and state["horizons"] == c.HORIZONS
    )


def probe_seconds(path):
    probe = json.loads(path.read_text())
    embedding, full = probe["embedding"], probe["full"]
    if any(not math.isfinite(value) or value <= 0 for value in (embedding, full)):
        raise RuntimeError(f"Invalid training probe timings: {path}")
    return embedding * c.STAGE_STEPS + full * (c.TRAIN_STEPS - c.STAGE_STEPS)


def evaluation_record(exp, checkpoint_sha256):
    root = c.ROOT / "runs/out" / exp.name
    record = json.loads((root / "evaluated.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if record["experiment"] != vars(exp) or manifest["experiment"] != vars(exp):
        raise RuntimeError(f"Evaluation used another experiment: {root}")
    recorded_checksum = manifest["checkpoint_sha256"]
    if (
        recorded_checksum != checkpoint_sha256
        or record.get("checkpoint_sha256", recorded_checksum) != checkpoint_sha256
    ):
        raise RuntimeError(f"Evaluation used another checkpoint: {root}")
    return record, manifest


def data(split):
    root = c.SOURCE_ROOT / "runs/data"
    manifest = json.loads((root / "manifest.json").read_text())
    path = root / f"{split}.jsonl"
    if (
        manifest["design_signature"] != c.SOURCE_SIGNATURE
        or manifest["model"] != c.MODEL
        or manifest["revision"] != c.REVISION
        or file_hash(path) != manifest["sha256"][split]
    ):
        raise RuntimeError("Instruction data provenance mismatch")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) != manifest["rows"][split]:
        raise RuntimeError("Instruction data count mismatch")
    return records


def load_model(device, backend=c.BACKEND):
    import torch

    from scripts.instruct_common import load_model as original
    from scripts.vllm_sps_model import QwenSPSMTP

    state = torch.load(c.PARENT, map_location="cpu", mmap=True, weights_only=False)
    if (
        state["source"] != c.MODEL
        or state["revision"] != c.REVISION
        or state["design_signature"] != c.PARENT_SIGNATURE
        or state["step"] != c.PARENT_STEP
    ):
        raise RuntimeError("SPS parent provenance mismatch")
    model = original("staged_sps", "cpu")
    model.load_state_dict(state["model"])
    del state
    return QwenSPSMTP(model, c.HORIZONS, backend=backend).to(device)


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
        sps_parent_signature=c.PARENT_SIGNATURE,
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
    AutoTokenizer.from_pretrained(c.MODEL, revision=c.REVISION).save_pretrained(
        destination
    )
