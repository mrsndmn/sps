"""Compare a trained DFlash-b16 checkpoint with published b16 and sequential AR."""

import json

import torch

from experiments.environment import HF_HOME
from workloads.distillation import dflash_mask_ablation as reference, dflash_repro_train as training
from orchestration import mlspace_launch as launch
from workloads.distillation import qwen4b_drafter_eval as paired
from workloads.distillation import qwen4b_drafter_train as previous

from hashlib import sha256


def settings_digest(settings):
    return sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def signature():
    return "gsm8k"


def gsm8k(args):
    from huggingface_hub import hf_hub_download
    from pyarrow.parquet import read_table

    path = hf_hub_download(args.gsm8k, args.gsm8k_file, repo_type="dataset", revision=args.gsm8k_revision,
                           cache_dir=HF_HOME / "hub", local_files_only=True)
    return read_table(path, columns=["question"])["question"].to_pylist()


@torch.inference_mode()
def evaluate(args):
    from safetensors.torch import load_file
    from transformers import AutoModel

    checkpoint = args.checkpoint
    inputs = json.loads((checkpoint / "inputs.json").read_text())
    if settings_digest(inputs.pop("settings")) != args.training_settings_sha256 or inputs != training.input_provenance():
        raise ValueError("DFlash checkpoint training inputs do not match this experiment")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("The GSM8K benchmark requires exactly one visible GPU")
    device = torch.device("cuda:0")
    target, tokenizer = previous.load_target(device)
    eos = target.generation_config.eos_token_id
    stops = {eos} if isinstance(eos, int) else set(eos)
    spec = reference.Experiment("bidirectional", args.proposals)
    trained = training.build_draft(device, experiment=args).to(torch.bfloat16).eval()
    trained.load_state_dict(load_file(checkpoint / "weights.safetensors"), strict=True)
    published = AutoModel.from_pretrained(
        reference.DRAFT, revision=reference.DRAFT_REVISION, code_revision=reference.DRAFT_REVISION,
        dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True, local_files_only=True,
    ).to(device).eval()
    questions = gsm8k(args)
    order = [("ar", None), ("published", published), ("trained", trained)]
    samples = {name: [] for name, _ in order}

    def prompt(question):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=True, return_tensors="pt",
            add_generation_prompt=True, enable_thinking=False,
        ).to(device)

    def decode(draft, ids, limit):
        if draft is None:
            return paired.ar_decode(target, ids, stops, limit)
        return reference.generate(target, draft, ids, spec, stops, limit)

    warmup = prompt(questions[0])
    for _, draft in order:
        decode(draft, warmup, args.warmup_tokens)
    for index, question in enumerate(questions):
        ids = prompt(question)
        offset = index % len(order)
        runs = {name: decode(draft, ids, args.limit) for name, draft in order[offset:] + order[:offset]}
        for name, row in runs.items():
            row.update(index=index, repeat=0, greedy_match=row["tokens"] == runs["ar"]["tokens"])
            samples[name].append(row)
        if (index + 1) % 25 == 0:
            print(f"args.gsm8k {index + 1}/{len(questions)}", flush=True)
    summary = {name: reference.summary(samples[name], spec) for name in ("published", "trained")}
    summary["ar"] = {"requests": len(questions), "tokens_per_second": paired.decode_rate(samples["ar"])}
    summary["all_arms_matched"] = paired.matched_throughput(samples)
    summary["draft_pair"] = paired.compare_drafts(samples["published"], samples["trained"])
    launch.store_json(args.result, dict(
        design=args.name, evaluation=signature(), checkpoint=str(checkpoint),
        target=reference.TARGET_REVISION,
        published=reference.DRAFT_REVISION, gsm8k=args.gsm8k_revision,
        gpu=torch.cuda.get_device_name(), summary=summary, samples=samples,
    ))
    print(json.dumps(summary, indent=2), flush=True)
