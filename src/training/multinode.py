"""Check the MLSpace rank layout and exercise DDP-sized messages before training."""
from collections import defaultdict
import json
import os
from pathlib import Path
import socket
import time

import torch
from torch import distributed as dist


def save_report(out_dir, **report):
    if os.environ["RANK"] == "0":
        path = Path(out_dir) / "interconnect.json"
        temp = path.with_suffix(".tmp")
        report["master_addr"] = os.environ["MASTER_ADDR"]
        temp.write_text(json.dumps(report))
        temp.replace(path)


def check_interconnect(out_dir, benchmark):
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    nodes = int(os.environ["SPS_MPI_NODES"])
    per_node = world // nodes
    placements = [None] * world
    save_report(out_dir, stage="topology_gather")
    dist.all_gather_object(placements, (socket.gethostname(), local, torch.cuda.device_count()))
    hosts = defaultdict(list)
    for host, local_rank, visible in placements:
        if visible != per_node:
            raise RuntimeError(f"Expected {per_node} visible GPUs on {host}, got {visible}")
        hosts[host].append(local_rank)
    if len(hosts) != nodes or any(sorted(ranks) != list(range(per_node)) for ranks in hosts.values()):
        raise RuntimeError(f"Unexpected MPI placement: {dict(hosts)}")
    report = dict(rank_placements=placements, all_reduce_seconds_by_mib={})
    save_report(out_dir, stage="topology_verified", **report)
    if benchmark:
        for mib in (4, 25, 64):
            report["active_mib"] = mib
            save_report(out_dir, stage="all_reduce", **report)
            tensor = torch.empty(mib * 1024**2 // 4, device=f"cuda:{local}")
            samples = []
            for _ in range(3):
                tensor.fill_(rank + 1)
                torch.cuda.synchronize()
                start = time.perf_counter()
                dist.all_reduce(tensor)
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - start)
                if not (tensor == world * (world + 1) // 2).all().item():
                    raise RuntimeError(f"Invalid {mib} MiB all-reduce result")
            report["all_reduce_seconds_by_mib"][mib] = samples
            del report["active_mib"]
            save_report(out_dir, stage="all_reduce", **report)
    save_report(out_dir, stage="complete", **report)
    if rank == 0:
        print(f"MPI layout and NCCL checks: {report}", flush=True)
