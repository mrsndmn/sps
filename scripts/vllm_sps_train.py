"""Assistant-masked staged MTP adaptation with one Qwen backbone."""

import argparse
import contextlib
import json
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from scripts import instruct_config as source
from scripts import vllm_sps_common as common
from scripts import vllm_sps_config as c
from scripts.mlspace_launch import store_json
from scripts.vllm_sps_model import horizon_targets


def labels_for(row, device):
    # Fixed physical shape prevents recompiling FlexAttention per dialog length.
    padding = source.CONTEXT - len(row["input_ids"])
    if padding < 0:
        raise ValueError("Instruction row exceeds the pinned training context")
    tokens = torch.tensor(row["input_ids"] + [0] * padding, device=device)[None]
    mask = torch.tensor(row["loss_mask"] + [0] * padding, device=device)[None]
    return tokens[:, :-1], horizon_targets(tokens, mask, c.HORIZONS)


@torch.no_grad()
def validation(model, rows, device):
    model.eval()
    total = 0.0
    count = 0
    for row in rows:
        tokens, labels = labels_for(row, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, n = model(tokens, targets=labels)
        total += loss.item()
        count += n
    model.train()
    return total / count


def optimizer_for(model, frozen):
    model.set_stage(frozen)
    if not frozen:
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=source.EMBED_LR if frozen else source.FULL_LR,
        betas=source.BETAS,
        weight_decay=0.0 if frozen else source.WEIGHT_DECAY,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    if world != c.experiment_configs[0].num_gpus or c.GLOBAL_BATCH % world:
        raise ValueError("Unexpected training allocation")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl")
    torch.manual_seed(c.SEED)
    data = common.data("train")
    heldout = common.data("val")[: c.VAL_ROWS]
    if not {x["mode"] for x in heldout} == {"thinking", "nonthinking"}:
        raise RuntimeError("Validation must contain both generation modes")
    steps = c.PROBE_STEPS if args.probe else c.TRAIN_STEPS
    transition = steps // 2 if args.probe else c.STAGE_STEPS
    if len(data) < steps * c.GLOBAL_BATCH:
        raise RuntimeError("Insufficient instruction training data")
    root = c.ROOT / ("sanity" if args.probe else "runs") / "out" / "qwen3_sps_mtp"
    root.mkdir(parents=True, exist_ok=True)
    signature = common.signature()
    model = common.load_model(device)
    start = 0
    best = float("inf")
    state = None
    if not args.probe and (root / "latest.pt").exists():
        state = torch.load(
            root / "latest.pt", map_location="cpu", mmap=True, weights_only=False
        )
        if state["design_signature"] != signature or state["parent"] != str(c.PARENT):
            raise RuntimeError("Resume provenance mismatch")
        model.load_state_dict(state["model"])
        start, best = state["step"], state["best"]
    frozen = start < transition
    optimizer = optimizer_for(model, frozen)
    if state is not None and state["frozen"] == frozen:
        optimizer.load_state_dict(state["optimizer"])
    state = None
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False)
    accumulation = c.GLOBAL_BATCH // world
    times = {"embedding": [], "full": []}

    def save(path, step):
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "best": best,
                "frozen": frozen,
                "parent": str(c.PARENT),
                "design_signature": signature,
                "horizons": c.HORIZONS,
                "source": c.MODEL,
                "revision": c.REVISION,
            },
            temporary,
        )
        temporary.replace(path)

    for step in range(start, steps):
        if step == transition:
            del ddp
            frozen = False
            optimizer = optimizer_for(model, frozen)
            ddp = DDP(model, device_ids=[local], broadcast_buffers=False)
            best = float("inf")
        optimizer.zero_grad(set_to_none=True)
        batch = [
            labels_for(data[step * c.GLOBAL_BATCH + j * world + rank], device)
            for j in range(accumulation)
        ]
        denominator = sum(
            int((labels[:, :, 1:] != -100).sum())
            if frozen
            else int((labels != -100).sum())
            for _, labels in batch
        )
        denominator = torch.tensor(denominator, dtype=torch.float64, device=device)
        dist.all_reduce(denominator)
        if denominator <= 0:
            raise RuntimeError("No supervised horizon targets")
        torch.cuda.synchronize()
        began = time.monotonic()
        for j, (tokens, labels) in enumerate(batch):
            sync = contextlib.nullcontext() if j == accumulation - 1 else ddp.no_sync()
            with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = ddp(tokens, targets=labels)
                (loss * (world / denominator)).backward()
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], source.CLIP
        )
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite MTP gradient")
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - began
        phase = "embedding" if frozen else "full"
        # Exclude first compilation/initialization update of each phase.
        if step not in {0, transition}:
            times[phase].append(elapsed)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "phase": phase,
                        "seconds": elapsed,
                        "gradient_norm": norm.item(),
                    }
                ),
                flush=True,
            )
        milestone = (step + 1) % c.CHECKPOINT_INTERVAL == 0 or step + 1 in {
            transition,
            steps,
        }
        if milestone and not args.probe:
            dist.barrier()
            if rank == 0:
                score = validation(model, heldout, device)
                if score < best:
                    best = score
                    save(
                        root / ("embedding_best.pt" if frozen else "best.pt"), step + 1
                    )
                save(root / "latest.pt", step + 1)
                with (root / "metrics.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "step": step + 1,
                                "val_horizon_ce": score,
                                "frozen": frozen,
                            }
                        )
                        + "\n"
                    )
            dist.barrier()
    if args.probe:
        measured = torch.tensor(
            [max(times[k], default=0.0) for k in times], device=device
        )
        dist.all_reduce(measured, op=dist.ReduceOp.MAX)
        if rank == 0:
            probe = {
                "design_signature": signature,
                "embedding": measured[0].item(),
                "full": measured[1].item(),
                "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
            }
            store_json(root / "probe.json", probe)
            store_json(c.ROOT / "prepared.json", probe)
    elif rank == 0:
        save(root / "final.pt", steps)
        # Export the final checkpoint; ordinary and speculative SPS share it.
        common.export(model, root / "hf")
        store_json(
            root / "trained.json",
            {
                "design_signature": signature,
                "checkpoint": str(root / "final.pt"),
                "export": str(root / "hf"),
                "parent": str(c.PARENT),
                "parent_sha256": common.file_hash(c.PARENT),
                "checkpoint_sha256": common.file_hash(root / "final.pt"),
                "steps": steps,
                "horizons": c.HORIZONS,
            },
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
