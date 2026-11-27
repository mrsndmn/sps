"""Distribute the deterministic validation stream without changing its batches."""

import torch
from torch import distributed as dist


def validation_batches(num_batches, batch_size, num_starts, rank, world_size):
    rng = torch.Generator().manual_seed(42)
    for index in range(num_batches):
        starts = torch.randint(num_starts, (batch_size,), generator=rng)
        if index % world_size == rank:
            yield index, starts


def collect_validation(local, world_size):
    """Reassemble per-batch loss and sufficient statistics in serial order."""
    parts = [local]
    if world_size > 1:
        parts = [None] * world_size
        dist.all_gather_object(parts, local)
    losses, sums, occurrences = [], {}, {}
    for _, loss, batch_sums, batch_occurrences in sorted(row for part in parts for row in part):
        losses.append(loss)
        for key, value in batch_sums.items():
            sums[key] = sums.get(key, 0.0) + value
            occurrences[key] = occurrences.get(key, 0) + batch_occurrences[key]
    return torch.tensor(losses, dtype=torch.float32).mean(), sums, occurrences
