"""Train future slots, then jointly adapt the native AR readout and slots."""
import argparse
import contextlib
import json
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from scripts import instruct_config as source
from scripts import real_readout_run as run
from scripts import vllm_sps_common as common
from scripts import vllm_sps_config as c
from scripts.instruct_common import load_model
from scripts.mlspace_launch import store_json
from scripts.real_readout_model import RealReadoutMTP
from scripts.vllm_sps_train import labels_for, matches_schedule, optimizer_for, restore_optimizer, validation


def parent():
    state = torch.load(run.PARENT, map_location="cpu", mmap=True, weights_only=False)
    if (state["design_signature"] != c.PARENT_SIGNATURE or state["step"] != c.PARENT_STEP
            or state["source"] != c.MODEL or state["revision"] != c.REVISION
            or state["experiment"]["name"] != run.PARENT.parent.name
            or state["experiment"]["recipe"] != "continued_ar" or state["experiment"]["distillation_weight"]):
        raise ValueError("Unexpected AR parent")
    wrapper = load_model("continued_ar", "cpu")
    wrapper.load_state_dict(state["model"])
    return RealReadoutMTP(wrapper.backbone, c.HORIZONS, c.BACKEND)


@torch.no_grad()
def audit_readout(model, example, device):
    ids = torch.tensor([example["input_ids"][:64]], device=device)
    model.eval()
    model.backbone.set_attn_implementation("sdpa")
    model.backend = "sdpa"
    try:
        reference = model.backbone(ids, use_cache=False).logits
        observed = model(ids).logits[:, :, 0]
        torch.testing.assert_close(observed, reference, atol=2e-4, rtol=2e-4)
        fp32_error = float((observed-reference).abs().max())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            native_bf16 = model.backbone(ids, use_cache=False).logits.float()
            model.backend = c.BACKEND
            model.backbone.set_attn_implementation("sps_flex_attention")
            packed_bf16 = model(ids).logits[:, :, 0].float()
        tv = 0.5 * (packed_bf16.softmax(-1) - native_bf16.softmax(-1)).abs().sum(-1)
        # Previous BF16 layout controls reached TV 0.0312. Use 0.10 as a
        # gross-regression gate and retain measured values for interpretation.
        if not bool((tv <= 0.10).all()):
            raise RuntimeError(f"Packed BF16 next-token distribution drift: {tv.max().item()}")
        return dict(fp32_max_abs=fp32_error, bf16_max_tv=tv.max().item(),
                    bf16_mean_tv=tv.mean().item(),
                    bf16_argmax_differences=int((packed_bf16.argmax(-1) != native_bf16.argmax(-1)).sum()))
    finally:
        model.backend = c.BACKEND
        model.backbone.set_attn_implementation("sps_flex_attention")
        model.train()


def atomic_save(path, record):
    temporary = path.with_suffix(".tmp")
    torch.save(record, temporary)
    temporary.replace(path)


def resume_audit(state, split):
    if state is None:
        return {}
    audit = state["readout_audit"]
    required = {"initial", "after_frozen"} if state["step"] >= split else {"initial"}
    if not required.issubset(audit):
        raise ValueError("Missing readout audit for resumed stage")
    return dict(audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("probe", "train"))
    probe = parser.parse_args().phase == "probe"
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != run.ARM.num_gpus or c.GLOBAL_BATCH % world:
        raise RuntimeError("Unexpected allocation")
    torch.manual_seed(c.SEED)
    data = common.data("train")
    heldout = common.data("val")[:c.VAL_ROWS]
    steps = c.PROBE_STEPS if probe else c.TRAIN_STEPS
    split = steps // 2 if probe else c.STAGE_STEPS
    model = parent().to(device)
    root = run.ROOT / "probe" if probe else run.OUT
    root.mkdir(parents=True, exist_ok=True)
    start = 0
    state = None
    latest = root / "latest.pt"
    if not probe and latest.exists():
        state = torch.load(latest, map_location="cpu", mmap=True, weights_only=False)
        if state["parent"] != str(run.PARENT) or not matches_schedule(state, steps, split):
            raise ValueError("Resume checkpoint does not match the configured run")
        model.load_state_dict(state["model"])
        start = state["step"]
    audit = resume_audit(state, split)
    if start == 0:
        audit["initial"] = audit_readout(model, heldout[0], device)
    durations = {}
    for frozen, begin, end in ((True, 0, split), (False, split, steps)):
        if start >= end:
            continue
        model.train()
        optimizer = optimizer_for(model, frozen)
        if state is not None and state["frozen"] == frozen:
            restore_optimizer(optimizer, state["optimizer"])
        state = None
        ddp = DDP(model, device_ids=[device.index], broadcast_buffers=False)
        times = []
        for step in range(max(begin, start), end):
            offset = step * c.GLOBAL_BATCH
            batch = [labels_for(data[offset+j], device)
                     for j in range(rank, c.GLOBAL_BATCH, world)]
            count = sum((labels[:, :, int(frozen):] != -100).sum() for _, labels in batch)
            dist.all_reduce(count)
            if count.item() <= 0:
                raise ValueError("Empty supervised batch")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            began = time.monotonic()
            for j, (ids, labels) in enumerate(batch):
                sync = contextlib.nullcontext() if j == len(batch)-1 else ddp.no_sync()
                with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = ddp(ids, targets=labels)
                    (loss * (world / count.double())).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), source.CLIP, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.monotonic() - began
            if step > begin:
                times.append(elapsed)
            if rank == 0:
                print(json.dumps(dict(step=step+1, frozen=frozen, seconds=elapsed)), flush=True)
            if frozen and step+1 == end:
                audit["after_frozen"] = audit_readout(model, heldout[0], device)
                if rank == 0:
                    store_json(root / "readout_audit.json", audit)
            if not probe and ((step+1) % c.CHECKPOINT_INTERVAL == 0 or step+1 == end):
                dist.barrier()
                if rank == 0:
                    score = validation(model, heldout, device)
                    record = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                  step=step+1, frozen=frozen, stage_steps=split,
                                  global_batch=c.GLOBAL_BATCH, training_steps=steps,
                                  parent=str(run.PARENT), readout_audit=dict(audit), val_horizon_ce=score)
                    if step+1 == end:
                        atomic_save(root / ("stage_endpoint.pt" if frozen else "final.pt"), record)
                    atomic_save(latest, record)
                    with (root / "metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps(dict(step=step+1, val_horizon_ce=score))+"\n")
                dist.barrier()
        durations[frozen] = max(times, default=0.)
        del ddp, optimizer
    if probe:
        measured = torch.tensor([durations[True], durations[False]], device=device)
        dist.all_reduce(measured, op=dist.ReduceOp.MAX)
        if rank == 0:
            projected = (measured[0]*c.STAGE_STEPS + measured[1]*(c.TRAIN_STEPS-c.STAGE_STEPS))/3600
            store_json(run.ROOT / "prepared.json", dict(
                       **common.probe_workload(), seconds=measured.tolist(), projected_hours=projected.item()))
    elif rank == 0:
        store_json(root / "trained.json", dict(parent=str(run.PARENT),
                   checkpoint=str(root/"final.pt"), step=steps,
                   checkpoint_sha256=common.file_hash(root/"final.pt")))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
