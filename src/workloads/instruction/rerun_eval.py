"""One-GPU checkpoint/workload evaluation, independent of long sibling tasks."""

import importlib.metadata
import json
import time

from orchestration.mlspace_launch import store_json


def evaluate(sc, c, variant, exp, workload, limit=None):
    import torch
    from datasets import load_dataset
    from langdetect import DetectorFactory
    from transformers import AutoTokenizer

    from experiments import instruct_config as source
    from workloads.instruction.instruct_common import load_model, nll
    from data.instruct_data import rows
    from workloads.instruction.instruct_evaluate import (
        ensure_ifeval_resources,
        evaluate_generation,
        likelihood,
    )

    sc.source_plan()
    checkpoint = sc.validate_checkpoint(variant, exp)
    root = sc.output_dir(variant, exp, workload) / ("smoke" if limit else "eval")
    root.mkdir(parents=True, exist_ok=True)
    gate = json.loads((sc.e.SANITY_ROOT / "generation-protocol.json").read_text())
    if (
        gate['generation'] != source.family_config.GENERATION
    ):
        raise RuntimeError("Source generation protocol mismatch")
    DetectorFactory.seed = source.family_config.SEED
    ensure_ifeval_resources()
    original = json.loads((sc.e.output_dir(exp) / "eval/manifest.json").read_text())
    if (
        original['lm_eval_version'] != importlib.metadata.version('lm_eval')
    ):
        raise RuntimeError("Scoring implementation differs from source evaluation")
    manifest = {
        "design_signature": sc.signature(),
        "source_suite": c.SOURCE_SUITE,
        "experiment": exp.__dict__,
        "checkpoint": checkpoint,
        "workload": workload,
        "limit": limit,
        "generation": source.family_config.GENERATION,
        "lm_eval_version": original["lm_eval_version"],
        "datasets": {
            source.family_config.IFEVAL_DATASET: source.family_config.IFEVAL_REVISION,
            source.family_config.MMLU_DATASET: source.family_config.MMLU_REVISION,
        },
    }
    meta = root / "manifest.json"
    if meta.exists() and json.loads(meta.read_text()) != manifest:
        raise RuntimeError("Evaluation resume provenance mismatch")
    store_json(meta, manifest)
    model = load_model(exp.recipe, "cuda").eval().requires_grad_(False)
    state = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    del state
    tokenizer = AutoTokenizer.from_pretrained(source.family_config.MODEL, revision=source.family_config.REVISION)
    result = {
        "design_signature": sc.signature(),
        "experiment": exp.name,
        "checkpoint": checkpoint,
        "workload": workload,
    }
    started = time.monotonic()
    if workload == "quality":
        data = rows("test")
        result["assistant_nll"] = nll(
            model, data[:limit] if limit else data, torch.device("cuda")
        )
        store_json(root / "heldout.json", result["assistant_nll"])
        likelihood(model, tokenizer, root, limit)
    else:
        task, mode = workload.split("-", 1)
        if task == "ifeval":
            data = list(
                load_dataset(
                    source.family_config.IFEVAL_DATASET,
                    revision=source.family_config.IFEVAL_REVISION,
                    split="train",
                )
            )
        else:
            data = list(
                load_dataset(
                    source.family_config.MMLU_DATASET,
                    "all",
                    revision=source.family_config.MMLU_REVISION,
                    split="test",
                )
            )
        if limit:
            data = data[:limit]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            result["metrics"] = evaluate_generation(
                model,
                tokenizer,
                data,
                task,
                mode,
                source.family_config.GENERATION[mode],
                root,
                sc.signature(),
                source.family_config.SEED,
                started,
                c.WORKLOAD_HOURS[workload] - c.STARTUP_SECONDS / 3600,
            )
    store_json(root / "results.json", result)
    if not limit:
        store_json(sc.output_dir(variant, exp, workload) / "evaluated.json", result)
    print(json.dumps(result), flush=True)
