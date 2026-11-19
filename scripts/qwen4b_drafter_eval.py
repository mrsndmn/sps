"""Evaluate trained block drafters with one frozen verifier on a single GPU."""

from hashlib import sha256
import json
from pathlib import Path
import time

import torch

from scripts import dflash_mask_ablation as reference
from scripts import instruct_config, mlspace_launch as launch
from scripts import qwen4b_drafter_train as training

PROMPTS = 64
REPEATS = 2
LIMIT = 128
WARMUP_TOKENS = 24


def signature():
    digest = sha256(training.signature().encode())
    for path in (Path(__file__), Path(reference.__file__), Path(instruct_config.__file__),
                 Path(reference.select_examples.__code__.co_filename),
                 Path(reference.mmlu_messages.__code__.co_filename)):
        digest.update(sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def ar_decode(target, ids, stops, limit):
    from transformers import DynamicCache

    cache = DynamicCache()
    logits = target(ids, past_key_values=cache, use_cache=True, logits_to_keep=1).logits
    token = logits[0, -1].argmax().item()
    tokens = [token]
    torch.cuda.synchronize(ids.device)
    start = time.perf_counter()
    while token not in stops and len(tokens) < limit:
        logits = target(
            ids.new_tensor([[token]]), past_key_values=cache, use_cache=True, logits_to_keep=1,
        ).logits
        token = logits[0, -1].argmax().item()
        tokens.append(token)
    torch.cuda.synchronize(ids.device)
    return dict(tokens=tokens, decode_seconds=time.perf_counter() - start)


def validate_checkpoints(train_root):
    metadata = {}
    for exp in training.experiment_configs:
        root = train_root / exp.name
        info = json.loads((root / "trained.json").read_text())
        if info["design_signature"] != training.signature() or info["experiment"] != exp.name:
            raise RuntimeError(f"Unexpected checkpoint for {exp.name}")
        if info["steps"] != training.EPOCHS * training.ROWS["train"] // training.ACCUMULATION:
            raise RuntimeError(f"Incomplete checkpoint for {exp.name}")
        if not (root / "weights.safetensors").is_file():
            raise RuntimeError(f"Missing weights for {exp.name}")
        metadata[exp.name] = info
    if len({info["initial_sha256"] for info in metadata.values()}) != 1:
        raise RuntimeError("The four Transformer initializations differ")
    return metadata


@torch.inference_mode()
def evaluate(train_root, result):
    import transformers
    from datasets import load_dataset
    from safetensors.torch import load_file

    if torch.cuda.device_count() != 1:
        raise RuntimeError("The benchmark requires one visible GPU")
    metadata = validate_checkpoints(train_root)
    device = torch.device("cuda:0")
    target, tokenizer = training.load_target(device)
    eos = target.generation_config.eos_token_id
    stops = {eos} if isinstance(eos, int) else set(eos)
    drafters = {}
    for exp in training.experiment_configs:
        draft = training.build_drafter(target, exp, device).to(torch.bfloat16)
        weights = load_file(train_root / exp.name / "weights.safetensors")
        draft.load_state_dict(weights, strict=True)
        drafters[exp.name] = draft.eval()
    conditions = [
        (exp, reference.Experiment(exp.mask, width))
        for width in training.LENGTHS
        for exp in training.experiment_configs
    ]
    dataset = load_dataset(
        instruct_config.MMLU_DATASET, "all", revision=instruct_config.MMLU_REVISION, split="test",
    )
    indices = reference.select_examples(dataset)[: PROMPTS + 1]
    if len(indices) != PROMPTS + 1:
        raise RuntimeError("Not enough benchmark prompts")

    def prompt(index):
        return tokenizer.apply_chat_template(
            reference.mmlu_messages(dataset[index]), tokenize=True,
            return_tensors="pt", add_generation_prompt=True, enable_thinking=False,
        ).to(device)

    warmup = prompt(indices[0])
    ar_decode(target, warmup, stops, WARMUP_TOKENS)
    for exp, spec in conditions:
        reference.generate(target, drafters[exp.name], warmup, spec, stops, WARMUP_TOKENS)

    specs = {f"{exp.name}-{spec.draft_tokens}": spec for exp, spec in conditions}
    order = [("ar", None, None)] + [
        (f"{exp.name}-{spec.draft_tokens}", drafters[exp.name], spec)
        for exp, spec in conditions
    ]
    samples = {name: [] for name, _, _ in order}
    inputs = {}
    for offset, index in enumerate(indices[1:]):
        ids = prompt(index)
        inputs[index] = ids[0].tolist()
        for repeat in range(REPEATS):
            rotation = (offset + repeat) % len(order)
            runs = {}
            for name, draft, spec in order[rotation:] + order[:rotation]:
                runs[name] = (
                    ar_decode(target, ids, stops, LIMIT) if spec is None
                    else reference.generate(target, draft, ids, spec, stops, LIMIT)
                )
            for name, sample in runs.items():
                sample.update(index=index, repeat=repeat,
                              greedy_match=sample["tokens"] == runs["ar"]["tokens"])
                samples[name].append(sample)
        print(f"Completed prompt {offset + 1}/{PROMPTS}", flush=True)

    summary = {name: reference.summary(samples[name], spec) for name, spec in specs.items()}
    ar = samples["ar"]
    baseline = sum(len(s["tokens"]) - 1 for s in ar) / sum(s["decode_seconds"] for s in ar)
    summary["ar"] = {"requests": len(ar), "tokens_per_second": baseline}
    for name in specs:
        summary[name]["speedup_vs_ar"] = summary[name]["tokens_per_second"] / baseline
    launch.store_json(result, dict(
        design_signature=signature(), gpu=torch.cuda.get_device_name(device),
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        target=[reference.TARGET, reference.TARGET_REVISION],
        dataset=[instruct_config.MMLU_DATASET, instruct_config.MMLU_REVISION],
        thinking=False, metadata=metadata, inputs=inputs, samples=samples, summary=summary,
    ))
    print(json.dumps(summary, indent=2), flush=True)
