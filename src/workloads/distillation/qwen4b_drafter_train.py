"""Train a five-layer block drafter on a frozen Qwen3-4B."""

from hashlib import sha256
from inspect import getsource
import json
from pathlib import Path
import random
from statistics import median
import time

import torch
from torch import nn
from torch.nn import functional as F

from experiments.environment import COMMON_ROOT
from workloads.distillation import dflash_mask_ablation as reference
from data import instruct_data
from orchestration import mlspace_launch as launch
from workloads.distillation import qwen4b_drafter_data as prepared
from experiments.vllm_sps_config import SOURCE_ROOT
from experiments.qwen4b_drafter_train import (
    ROOT,
    DATA,
    LENGTHS,
    TARGET_LAYERS,
    CONTEXT,
    DEFAULT_DATASET,
    DATASETS,
    ROWS,
    ANCHORS,
    EPOCHS,
    ACCUMULATION,
    LR,
    WARMUP,
    GAMMA,
    SEED,
    PROBE_STEPS,
    Experiment,
    experiment_configs,
)


def prepared_settings():
    return dict(context=CONTEXT, horizon=max(LENGTHS), validation=ROWS["val"],
                accumulation=ACCUMULATION, seed=SEED)


def selection_path():
    return prepared.selection_path(ROOT, **prepared_settings())


def selection():
    path = selection_path()
    if not path.is_file():
        raise RuntimeError(f"Run --dataset {prepared.NAME} --prepare-data before using {path}")
    return json.loads(path.read_text())


def training_rows(dataset):
    return ROWS["train"] if dataset == DEFAULT_DATASET else len(selection()["train"])


def signature(dataset=DEFAULT_DATASET):
    return dataset


class BlockDrafter(nn.Module):
    def __init__(self, draft, mask_embedding, embeddings):
        super().__init__()
        self.draft = draft
        count = 1 if embeddings == "shared" else max(LENGTHS)
        self.mask_embeddings = nn.Parameter(mask_embedding.detach().float().repeat(count, 1))
        self.target_layer_ids = draft.target_layer_ids
        self.mask_token_id = draft.mask_token_id

    def forward(self, *, noise_embedding, slots=None, **kwargs):
        if slots is None:
            slots = torch.arange(noise_embedding.size(1), device=noise_embedding.device)
        indices = (slots - 1).clamp(min=0, max=self.mask_embeddings.size(0) - 1)
        masks = self.mask_embeddings[indices].to(noise_embedding.dtype)
        inputs = torch.where(slots[None, :, None] == 0, noise_embedding, masks[None])
        return self.draft(noise_embedding=inputs, **kwargs)


def load_target(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    target = AutoModelForCausalLM.from_pretrained(
        reference.TARGET, revision=reference.TARGET_REVISION, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(
        reference.TARGET, revision=reference.TARGET_REVISION, local_files_only=True,
    )
    return target, tokenizer


def build_drafter(target, exp, device):
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    pinned = dict(revision=reference.DRAFT_REVISION, local_files_only=True)
    config = AutoConfig.from_pretrained(reference.DRAFT, **pinned)
    config._attn_implementation = "sdpa"
    cls = get_class_from_dynamic_module(
        config.auto_map["AutoModel"], reference.DRAFT, **pinned,
    )
    if config.num_hidden_layers != 5 or config.hidden_size != target.config.hidden_size:
        raise RuntimeError("Unexpected draft architecture")
    torch.manual_seed(SEED)
    draft = cls(config)
    if tuple(draft.target_layer_ids) != TARGET_LAYERS:
        raise RuntimeError("Unexpected target feature layers")
    embedding = target.get_input_embeddings().weight[draft.mask_token_id]
    return BlockDrafter(draft, embedding, exp.embeddings).to(device)


def candidate_anchors(loss_mask, length):
    """Keep the known token and all predictions inside the final response."""
    return [
        index for index in range(1, len(loss_mask) - length)
        if all(loss_mask[index : index + length + 1])
    ]


def read_rows(tokenizer, split, limit):
    manifest = json.loads((DATA / "manifest.json").read_text())
    path = DATA / f"{split}.jsonl"
    if instruct_data.file_hash(path) != manifest["sha256"][split]:
        raise RuntimeError(f"Source data hash changed for {split}")
    rows = []
    with path.open() as source:
        for line in source:
            record = json.loads(line)
            if record["mode"] != "nonthinking":
                continue
            try:
                row = instruct_data.encode_example(
                    tokenizer, record["messages"], "nonthinking", record["source_id"], CONTEXT,
                )
            except ValueError:
                continue
            for index, (token, assistant) in enumerate(zip(row["input_ids"], row["loss_mask"])):
                if assistant and token == tokenizer.eos_token_id:
                    for field in ("input_ids", "loss_mask"):
                        row[field] = row[field][: index + 1]
                    break
            if candidate_anchors(row["loss_mask"], max(LENGTHS)):
                rows.append(row)
            if len(rows) == limit:
                return rows
    raise RuntimeError(f"Only {len(rows)}/{limit} usable rows in {split}")


def block_batch(tokens, anchors, widths, mask):
    """Pack independent blocks after their shared source-token K/V columns."""
    layout = [(index, anchor, slot)
              for index, (anchor, width) in enumerate(zip(anchors, widths, strict=True))
              for slot in range(width + 1)]
    groups, starts, slots = tokens.new_tensor(layout).T
    source = torch.arange(tokens.size(1), device=tokens.device)
    block_positions = starts + slots
    same_group = groups[:, None] == groups[None, :]
    if mask == "causal":
        same_group &= slots[:, None] >= slots[None, :]
    visible = torch.cat((starts[:, None] > source[None, :], same_group), dim=-1)
    inputs = tokens[:, starts]
    positions = torch.cat((source, block_positions))[None]
    return inputs, positions, slots, visible[None, None], tokens[:, block_positions]


def sample_blocks(row, seed):
    rng = random.Random(seed)
    plans = []
    for width in LENGTHS:
        choices = candidate_anchors(row["loss_mask"], width)
        plans.extend((anchor, width) for anchor in rng.sample(
            choices, min(len(choices), ANCHORS // len(LENGTHS)),
        ))
    rng.shuffle(plans)
    return tuple(zip(*plans, strict=True))


def forward_loss(target, draft, row, exp, seed, device):
    tokens = torch.tensor([row["input_ids"]], device=device)
    anchors, widths = sample_blocks(row, seed)
    inputs, positions, slots, visible, labels = block_batch(tokens, anchors, widths, exp.mask)
    with torch.no_grad():
        output = target.model(tokens, use_cache=False, output_hidden_states=True)
        context = reference.context_features(output, TARGET_LAYERS)
        noise = target.get_input_embeddings()(inputs)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        hidden = draft(
            noise_embedding=noise, slots=slots, target_hidden=context,
            position_ids=positions, attention_mask=visible, is_causal=False,
        )
        predicted = slots > 0
        logits = target.lm_head(hidden[:, predicted])
        gold = labels[:, predicted]
        weights = torch.exp(-(slots[predicted].float() - 1) / GAMMA)
        per_slot = F.cross_entropy(logits.squeeze(0), gold.squeeze(0), reduction="none")
        loss = (weights * per_slot).sum() / weights.sum()
    return loss, (logits.argmax(-1) == gold).float().mean()


@torch.no_grad()
def validate(target, draft, rows, exp, device):
    draft.eval()
    metrics = [forward_loss(target, draft, row, exp, SEED + i, device)
               for i, row in enumerate(rows)]
    draft.train()
    loss, accuracy = torch.stack([torch.stack(pair) for pair in metrics]).mean(0).tolist()
    return {"loss": loss, "corpus_token_accuracy": accuracy}


def train(target, tokenizer, exp, root, *, probe=False, dataset=DEFAULT_DATASET):
    from transformers import get_cosine_schedule_with_warmup

    device = target.device
    if dataset == DEFAULT_DATASET:
        rows = read_rows(tokenizer, "train", PROBE_STEPS * ACCUMULATION if probe else ROWS["train"])
        val_rows = read_rows(tokenizer, "val", 4 if probe else ROWS["val"])
    else:
        corpus = prepared.open_dataset()
        splits = selection()
        train_indices, val_indices = splits["train"], splits["val"]
        if probe:
            train_indices = train_indices[: PROBE_STEPS * ACCUMULATION]
            val_indices = val_indices[:4]
        rows = corpus.select(train_indices)
        val_rows = corpus.select(val_indices)
    draft = build_drafter(target, exp, device)
    # Verify equality of the initial Transformer independently of mask cardinality.
    initial = sha256()
    for tensor in draft.draft.state_dict().values():
        initial.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    optimizer = torch.optim.AdamW(draft.parameters(), lr=LR, fused=True)
    steps = len(rows) // ACCUMULATION
    epochs = 1 if probe else EPOCHS
    schedule = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(steps * epochs * WARMUP)), steps * epochs,
    )
    history, durations = [], []
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(epochs):
        order = list(range(len(rows)))
        random.Random(SEED + epoch).shuffle(order)
        for step in range(steps):
            before = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for index in order[step * ACCUMULATION : (step + 1) * ACCUMULATION]:
                loss, _ = forward_loss(
                    target, draft, rows[index], exp, SEED + epoch * len(rows) + index, device,
                )
                (loss / ACCUMULATION).backward()
                losses.append(loss.detach())
            norm = nn.utils.clip_grad_norm_(draft.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            if probe or step % 25 == 0:
                print(f"{exp.name} epoch={epoch + 1} step={step + 1}/{steps} "
                      f"loss={torch.stack(losses).mean().item():.4f} norm={norm.item():.3f}",
                      flush=True)
                durations.append(time.perf_counter() - before)
        history.append(validate(target, draft, val_rows, exp, device))
        print(f"{exp.name} validation {history[-1]}", flush=True)
    report = dict(
        experiment=exp.name, dataset=dataset, design_signature=signature(dataset),
        initial_sha256=initial.hexdigest(),
        steps=steps * epochs, rows=len(rows), validation=history,
        seconds_per_probed_step=median(durations[1:]),
        peak_bytes=torch.cuda.max_memory_allocated(device),
    )
    if not probe:
        from safetensors.torch import save_file

        root.mkdir(parents=True, exist_ok=True)
        temporary = root / "weights.tmp"
        weights = {name: value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                   for name, value in draft.state_dict().items()}
        save_file(weights, temporary)
        temporary.replace(root / "weights.safetensors")
        launch.store_json(root / "trained.json", report)
    return report
