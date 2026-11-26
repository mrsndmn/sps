"""Paired zero-shot DFlash mask ablation on a frozen Qwen3-4B verifier."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import shlex
import time
from unittest.mock import patch

from experiments import COMMON_ROOT
from scripts import instruct_config as data_config
from scripts import mlspace_launch as launch
from scripts.instruct_generation import mmlu_messages
from scripts.vllm_sps_evaluate import select_examples

TARGET = "Qwen/Qwen3-4B"
TARGET_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
DRAFT = "z-lab/Qwen3-4B-DFlash-b16"
DRAFT_REVISION = "b74e3a329c4d963783143b1e970d95b002be72bd"
ROOT = COMMON_ROOT / "dflash-mask-ablation"
PROMPTS = 24
MAX_NEW_TOKENS = 128
REPEATS = 2


@dataclass(frozen=True)
class Experiment:
    mask: str
    draft_tokens: int

    @property
    def name(self):
        return f"{self.mask}-{self.draft_tokens}"


experiment_configs = [
    Experiment(mask, width) for width in (4, 8) for mask in ("bidirectional", "causal")
]


def signature():
    sources = [
        Path(__file__),
        Path(data_config.__file__),
        Path(mmlu_messages.__code__.co_filename),
        Path(select_examples.__code__.co_filename),
        Path(__file__).with_name("report_mmlu_stem.py"),
        Path(__file__).with_name("vllm_sps_config.py"),
    ]
    return sha256(
        b"".join(sha256(path.read_bytes()).digest() for path in sources)
    ).hexdigest()


def run_name():
    return signature()[:16]


def result_path():
    return ROOT / "runs" / "out" / run_name() / "evaluated.json"


def block_attention(positions, start, width, mode):
    """The rows are block queries and the columns are all cached and new keys."""
    import torch

    visible = torch.ones(
        (width, start + width), device=positions.device, dtype=torch.bool
    )
    if mode == "causal":
        visible = (
            positions[start : start + width, None] >= positions[None, : start + width]
        )
    return visible[None, None]


def commit_round(proposals, posterior, remaining, stop_ids):
    """Return the emitted suffix, accepted-prefix length, and termination flag."""
    accepted = 0
    for proposed, verified in zip(proposals, posterior):
        if proposed != verified:
            break
        accepted += 1
    suffix = proposals[:accepted] + [posterior[accepted]]
    for i, token in enumerate(suffix[:remaining]):
        if token in stop_ids or i + 1 == remaining:
            return suffix[: i + 1], accepted, True
    return suffix, accepted, False


def round_is_complete(suffix, accepted):
    return len(suffix) == accepted + 1


def context_features(output, target_layer_ids, committed=None):
    import torch

    return torch.cat(
        [output.hidden_states[layer + 1][:, :committed] for layer in target_layer_ids],
        dim=-1,
    )


@contextmanager
def cuda_timer(events):
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    events.append((start, end))


def generate(target, draft, ids, exp, stop_ids, limit):
    import torch
    from transformers import DynamicCache

    device = ids.device
    block_size = exp.draft_tokens + 1
    prefix_length = ids.size(1)
    positions = torch.arange(prefix_length + limit + block_size, device=device)
    kv_target = DynamicCache()
    kv_draft = DynamicCache()
    output = target(
        ids,
        past_key_values=kv_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    tokens = [output.logits[0, -1].argmax().item()]
    context = context_features(output, draft.target_layer_ids)
    block = torch.full(
        (1, block_size), draft.mask_token_id, device=device, dtype=ids.dtype
    )
    torch.cuda.synchronize(device)
    draft_events, verify_events, acceptance = [], [], []
    start_time = time.perf_counter()
    finished = tokens[0] in stop_ids or limit == 1
    while not finished:
        start = prefix_length + len(tokens) - 1
        end = start + block_size
        cached = kv_draft.get_seq_length()
        if kv_target.get_seq_length() != start or cached + context.size(1) != start:
            raise RuntimeError(
                "Draft features and target cache must end just before the anchor"
            )
        with cuda_timer(draft_events):
            block.fill_(draft.mask_token_id)
            block[0, 0] = tokens[-1]
            hidden = draft(
                noise_embedding=target.model.embed_tokens(block),
                target_hidden=context,
                position_ids=positions[None, cached:end],
                attention_mask=block_attention(positions, start, block_size, exp.mask),
                past_key_values=kv_draft,
                use_cache=True,
                is_causal=False,
            )
            kv_draft.crop(start)
            block[:, 1:] = target.lm_head(hidden[:, 1:]).argmax(dim=-1)
        with cuda_timer(verify_events):
            output = target(
                block,
                position_ids=positions[None, start:end],
                past_key_values=kv_target,
                use_cache=True,
                output_hidden_states=True,
            )
            comparison = torch.cat((block, output.logits.argmax(dim=-1)))
        guesses, posterior = comparison.tolist()
        suffix, accepted, finished = commit_round(
            guesses[1:], posterior, limit - len(tokens), stop_ids
        )
        tokens.extend(suffix)
        if round_is_complete(suffix, accepted):
            acceptance.append(accepted)
        if not finished:
            committed = accepted + 1
            kv_target.crop(start + committed)
            context = context_features(output, draft.target_layer_ids, committed)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    return {
        "tokens": tokens,
        "decode_seconds": elapsed,
        "acceptance": acceptance,
        "verify_rounds": len(verify_events),
        "draft_gpu_ms": sum(a.elapsed_time(b) for a, b in draft_events),
        "verify_gpu_ms": sum(a.elapsed_time(b) for a, b in verify_events),
    }


def summary(samples, exp):
    rounds = [accepted for sample in samples for accepted in sample["acceptance"]]
    if not rounds:
        raise ValueError(f"No complete verification rounds for {exp.name}")
    outputs = sum(len(sample["tokens"]) - 1 for sample in samples)
    steps = sum(sample["verify_rounds"] for sample in samples)
    return {
        "requests": len(samples),
        "greedy_matches": sum(sample["greedy_match"] for sample in samples),
        "tokens_per_second": outputs / sum(s["decode_seconds"] for s in samples),
        "tau": 1 + sum(rounds) / len(rounds),
        "acceptance": sum(rounds) / (len(rounds) * exp.draft_tokens),
        "acceptance_by_position": [
            sum(accepted >= pos for accepted in rounds) / len(rounds)
            for pos in range(1, exp.draft_tokens + 1)
        ],
        "complete_rounds": len(rounds),
        "verified_rounds": steps,
        "draft_gpu_ms_per_round": sum(s["draft_gpu_ms"] for s in samples) / steps,
        "verify_gpu_ms_per_round": sum(s["verify_gpu_ms"] for s in samples) / steps,
    }


def evaluate():
    import torch
    import transformers
    from datasets import load_dataset
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.device_count() != 1:
        raise RuntimeError("DFlash ablation requires exactly one visible GPU")
    device = torch.device("cuda:0")
    settings = dict(
        dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    )
    target = (
        AutoModelForCausalLM.from_pretrained(
            TARGET, revision=TARGET_REVISION, **settings
        )
        .to(device)
        .eval()
    )
    draft = (
        AutoModel.from_pretrained(
            DRAFT,
            revision=DRAFT_REVISION,
            code_revision=DRAFT_REVISION,
            trust_remote_code=True,
            **settings,
        )
        .to(device)
        .eval()
    )
    if draft.block_size != 16 or draft.target_layer_ids != [1, 9, 17, 25, 33]:
        raise RuntimeError("Unexpected DFlash-b16 checkpoint configuration")
    tokenizer = AutoTokenizer.from_pretrained(
        TARGET, revision=TARGET_REVISION, local_files_only=True
    )
    stop_ids = target.generation_config.eos_token_id
    stop_ids = {stop_ids} if isinstance(stop_ids, int) else set(stop_ids)
    dataset = load_dataset(
        data_config.MMLU_DATASET,
        "all",
        revision=data_config.MMLU_REVISION,
        split="test",
    )
    indices = select_examples(dataset)[: PROMPTS + 1]

    def input_tokens(index):
        return tokenizer.apply_chat_template(
            mmlu_messages(dataset[index]),
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=True,
        ).to(device)

    with torch.inference_mode():
        warmup = input_tokens(indices[0])
        for exp in experiment_configs:
            generate(target, draft, warmup, exp, stop_ids, 24)
        samples = {exp.name: [] for exp in experiment_configs}
        references = {}
        for offset, index in enumerate(indices[1:]):
            ids = input_tokens(index)
            # A plain greedy target run checks the actual speculative outputs.
            reference = target.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )[0, ids.size(1) :].tolist()
            references[index] = {"input": ids[0].tolist(), "output": reference}
            for repeat in range(REPEATS):
                rotation = (offset + repeat * 2) % len(experiment_configs)
                order = experiment_configs[rotation:] + experiment_configs[:rotation]
                for exp in order:
                    sample = generate(target, draft, ids, exp, stop_ids, MAX_NEW_TOKENS)
                    sample.update(
                        index=index,
                        repeat=repeat,
                        greedy_match=sample["tokens"] == reference,
                    )
                    samples[exp.name].append(sample)
            print(f"Completed prompt {offset + 1}/{PROMPTS}", flush=True)

    report = {
        "design_signature": signature(),
        "commit": launch.git("rev-parse", "HEAD"),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "target": [TARGET, TARGET_REVISION],
        "draft": [DRAFT, DRAFT_REVISION],
        "dataset": [data_config.MMLU_DATASET, data_config.MMLU_REVISION],
        "conditions": [asdict(exp) for exp in experiment_configs],
        "greedy_target": references,
        "samples": samples,
        "summary": {
            exp.name: summary(samples[exp.name], exp) for exp in experiment_configs
        },
    }
    launch.store_json(result_path(), report)
    print(json.dumps(report["summary"], indent=2))


def submit(dry):
    name = run_name()
    repo = Path(launch.git("rev-parse", "--show-toplevel"))
    commit = launch.git("rev-parse", "HEAD")
    staged = launch.STAGING_ROOT / commit
    body = launch.payload("eval", name, staged, "d.tarasov", "mrsndmn")
    body["job_desc"] = f"sps-dflash-mask-ablation {name} #d.tarasov @mrsndmn"
    body["script"] = f"cd {shlex.quote(str(staged))} && " + shlex.join(
        [str(launch.PYTHON), "-m", "scripts.dflash_mask_ablation", "--worker"]
    )
    if dry:
        print(json.dumps(body, indent=2))
        return

    from mls.manager.job.utils import training_job_api_from_profile

    with (
        patch.object(launch, "ARTIFACTS_ROOT", ROOT),
        patch.object(launch, "RUN_ROOT", ROOT / "runs"),
    ):
        with launch.submission_lock():
            if result_path().is_file():
                print(result_path())
                return
            client, options = training_job_api_from_profile("default")
            launch.disable_submission_retries(client)
            body["region"] = options["region"]
            active = {}
            for status in ("Pending", "Running"):
                response = client.get_list_jobs(
                    options["region"], None, status, 1000, 0
                )
                active.update(
                    {
                        launch.normalize(j["job_desc"]): j["job_name"]
                        for j in response["jobs"]
                    }
                )
            launch.reconcile_receipts(client, active)
            launch.reconcile_intents(active)
            identity = launch.normalize(body["job_desc"])
            if identity in active:
                print(
                    json.dumps(
                        {"name": name, "job_name": active[identity], "reused": True}
                    )
                )
                return
            launch.stage_repo(repo, commit)
            print(json.dumps(launch.submit_payload(client, body, "eval", name, commit)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for option in ("submit", "dry", "worker", "report"):
        mode.add_argument(f"--{option}", action="store_true")
    args = parser.parse_args()
    if args.worker:
        evaluate()
    elif args.report:
        print(json.dumps(json.loads(result_path().read_text())["summary"], indent=2))
    else:
        submit(args.dry)


if __name__ == "__main__":
    main()
