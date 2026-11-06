"""Sequential native-vLLM MMLU STEM quality and latency pilot."""

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

from scripts import instruct_config as source
from scripts import vllm_sps_common as common
from scripts import vllm_sps_config as c
from scripts.mlspace_launch import store_json


def select_examples(rows):
    from scripts.report_mmlu_stem import STEM_SUBJECTS

    groups = {}
    for index, row in enumerate(rows):
        if row["subject"] in STEM_SUBJECTS:
            groups.setdefault(row["subject"], []).append(index)
    rng = random.Random(c.SEED)
    for indices in groups.values():
        rng.shuffle(indices)
    selected = []
    while len(selected) < c.EVAL_COUNT:
        before = len(selected)
        for subject in sorted(groups):
            if groups[subject] and len(selected) < c.EVAL_COUNT:
                selected.append(groups[subject].pop())
        if len(selected) == before:
            raise RuntimeError("Insufficient MMLU STEM questions")
    return selected


def check_runtime():
    if sys.version_info < (3, 11):
        raise RuntimeError(
            "Pinned fresh vLLM main requires Python >=3.11; SPS environment is incompatible"
        )
    import torch

    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) != (2, 14):
        raise RuntimeError(
            "Pinned fresh vLLM main requires Torch 2.14 and matching compiled extensions"
        )
    import vllm

    source_root = Path(os.environ["SPS_VLLM_SOURCE"]).resolve()
    if source_root not in Path(vllm.__file__).resolve().parents:
        raise RuntimeError("Evaluation imported an unrelated vLLM installation")
    return torch, vllm


def engine_arguments(exp, export, eagle_revision):
    common_args = {
        "model": str(export) if exp.arm.startswith("sps") else c.MODEL,
        "revision": None if exp.arm.startswith("sps") else c.REVISION,
        "tokenizer": c.MODEL,
        "tokenizer_revision": c.REVISION,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_num_seqs": 1,
        "max_model_len": c.MAX_MODEL_LEN,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "enforce_eager": True,
        "async_scheduling": False,
        "attention_config": {"backend": "TRITON_ATTN"},
        "gpu_memory_utilization": c.GPU_MEMORY_UTILIZATION,
        "disable_log_stats": False,
        "seed": c.SEED,
    }
    if exp.arm == "sps_mtp":
        common_args["speculative_config"] = {
            "method": "custom_class",
            "model": "vllm.v1.spec_decode.sps_proposer.SPSProposer",
            "num_speculative_tokens": c.HORIZONS - 1,
        }
    elif exp.arm == "eagle3":
        if not eagle_revision:
            raise ValueError("A pinned EAGLE-3 revision is required")
        common_args["speculative_config"] = {
            "method": "eagle3",
            "model": c.EAGLE_MODEL,
            "revision": eagle_revision,
            "num_speculative_tokens": c.HORIZONS - 1,
        }
    return common_args


def evaluate(exp):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    torch, vllm = check_runtime()
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from scripts.instruct_generation import (
        document_seed,
        mmlu_messages,
        parse_answer,
        split_response,
    )

    signature = common.signature()
    trained = json.loads((c.ROOT / "runs/out/qwen3_sps_mtp/trained.json").read_text())
    if trained["design_signature"] != signature:
        raise RuntimeError("Training design does not match evaluation")
    checkpoint = Path(trained["checkpoint"])
    if common.file_hash(checkpoint) != trained["checkpoint_sha256"]:
        raise RuntimeError("Trained checkpoint changed")
    tokenizer = AutoTokenizer.from_pretrained(c.MODEL, revision=c.REVISION)
    rows = list(
        load_dataset(
            source.MMLU_DATASET, "all", revision=source.MMLU_REVISION, split="test"
        )
    )
    selected = select_examples(rows)
    # Resolve once per suite; immutable revision is shared by all arms/repeats.
    pin_path = c.ROOT / "eagle_revision.json"
    if pin_path.exists():
        eagle_revision = json.loads(pin_path.read_text())["revision"]
    else:
        eagle_revision = c.EAGLE_REVISION or HfApi().model_info(c.EAGLE_MODEL).sha
        # Concurrent writers must agree; use exclusive creation, not last writer wins.
        pin_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with pin_path.open("x") as stream:
                json.dump({"model": c.EAGLE_MODEL, "revision": eagle_revision}, stream)
        except FileExistsError:
            eagle_revision = json.loads(pin_path.read_text())["revision"]
    arguments = engine_arguments(exp, trained["export"], eagle_revision)
    references = None
    if exp.arm.startswith("sps"):
        from scripts.vllm_sps_audit import reference

        references = reference(trained)
    llm = LLM(**arguments)
    root = c.ROOT / "runs/out" / exp.name
    root.mkdir(parents=True, exist_ok=True)
    if references is not None:
        from scripts.vllm_sps_audit import verify

        audit = verify(llm, references)
        store_json(root / "greedy_audit.json", {"samples": audit})
        if not all(x["equal"] for x in audit):
            raise RuntimeError(
                "Native/vLLM SPS greedy parity failed; inspect greedy_audit.json before benchmarking"
            )
    manifest = {
        "design_signature": signature,
        "experiment": exp.__dict__,
        "indices": selected,
        "arguments": arguments,
        "checkpoint_sha256": trained["checkpoint_sha256"],
        "vllm_version": vllm.__version__,
        "vllm_source": os.environ["SPS_VLLM_SOURCE"],
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "dataset": source.MMLU_DATASET,
        "dataset_revision": source.MMLU_REVISION,
        "timer": "host request wall time; engine first/last token timestamps for decode",
        "draft_policy": "greedy SPS horizon proposals; target sampling uses matched settings",
    }
    store_json(root / "manifest.json", manifest)

    def generate(index, warmup=False):
        settings = dict(c.GENERATION[exp.mode])
        settings["max_tokens"] = settings.pop("max_new_tokens")
        if warmup:
            settings["max_tokens"] = min(settings["max_tokens"], 32)
        seed = document_seed(c.SEED, "mmlu", exp.mode, index)
        params = SamplingParams(**settings, seed=seed)
        ids = tokenizer.apply_chat_template(
            mmlu_messages(rows[index]),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=exp.mode == "thinking",
        )
        if len(ids) + settings["max_tokens"] > c.MAX_MODEL_LEN:
            raise RuntimeError("Prompt and output cap exceed benchmark context")
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = llm.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        answer = output.outputs[0]
        stats = output.metrics
        if stats is None or stats.last_token_ts < stats.first_token_ts:
            raise RuntimeError("vLLM did not return valid decode timing stats")
        decoded = split_response(
            tokenizer, list(answer.token_ids), exp.mode == "thinking"
        )
        parsed = parse_answer(decoded["final_text"])
        return dict(
            index=index,
            seed=seed,
            prompt_tokens=len(ids),
            output_tokens=len(answer.token_ids),
            token_ids=list(answer.token_ids),
            **decoded,
            answer=parsed,
            correct=parsed == "ABCD"[rows[index]["answer"]],
            subject=rows[index]["subject"],
            doc_hash=hashlib.sha256(
                json.dumps(rows[index], sort_keys=True).encode()
            ).hexdigest(),
            end_to_end_seconds=elapsed,
            decode_seconds=stats.last_token_ts - stats.first_token_ts,
            stop=answer.finish_reason,
            ttft_seconds=stats.first_token_latency,
        )

    for i in range(c.WARMUP_REQUESTS):
        generate(selected[i], warmup=True)
    samples = []
    # A failed timed arm is rerun wholly; partially warmed runs are never merged.
    with (root / "samples.jsonl").open("w") as stream:
        for repeat in range(c.REPEATS):
            for index in selected:
                item = dict(repeat=repeat, **generate(index))
                stream.write(json.dumps(item) + "\n")
                stream.flush()
                samples.append(item)
    result = []
    for repeat in range(c.REPEATS):
        trial = [x for x in samples if x["repeat"] == repeat]
        decode = sum(x["decode_seconds"] for x in trial)
        result.append(
            {
                "repeat": repeat,
                "count": len(trial),
                "accuracy": sum(x["correct"] for x in trial) / len(trial),
                "invalid_rate": sum(x["answer"] is None for x in trial) / len(trial),
                "length_rate": sum(x["stop"] == "length" for x in trial) / len(trial),
                "output_tokens": sum(x["output_tokens"] for x in trial),
                "end_to_end_seconds": sum(x["end_to_end_seconds"] for x in trial),
                "decode_seconds": decode,
                "decode_tokens_per_second": sum(
                    max(0, x["output_tokens"] - 1) for x in trial
                )
                / decode
                if decode
                else None,
            }
        )
    counters = []
    for metric in llm.get_metrics():
        if "spec_decode" in metric.name:
            counters.append(
                {
                    "name": metric.name,
                    "value": getattr(metric, "value", getattr(metric, "values", None)),
                }
            )
    store_json(
        root / "evaluated.json",
        {
            "design_signature": signature,
            "experiment": exp.__dict__,
            "trials": result,
            "acceptance_counters_including_warmup": counters,
        },
    )
