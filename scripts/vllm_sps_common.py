"""Pinned parent/data inputs and isolated SPS-MTP artifact identities."""

import hashlib
import json
from pathlib import Path

from scripts import vllm_sps_config as c


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature():
    repo = Path(__file__).resolve().parents[1]
    files = sorted((repo / "scripts").glob("vllm_sps_*.py"))
    files += [
        repo / "src/modeling/models/sps/qwen_mtp.py",
        repo / "src/modeling/models/sps/pretrained.py",
        repo / "scripts/instruct_common.py",
        repo / "scripts/instruct_config.py",
    ]
    return hashlib.sha256(
        json.dumps(
            {str(p.relative_to(repo)): file_hash(p) for p in files}, sort_keys=True
        ).encode()
    ).hexdigest()


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

    from modeling.models.sps.qwen_mtp import QwenSPSMTP
    from scripts.instruct_common import load_model as original

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
