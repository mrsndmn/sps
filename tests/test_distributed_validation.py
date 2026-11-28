import pytest

torch = pytest.importorskip("torch")
from torch import distributed as dist
from torch import multiprocessing as mp

from training.validation import collect_validation, validation_batches


def check_ranks(rank, rendezvous):
    world = 4
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world)
    try:
        for count in (2, 7):
            gen = torch.Generator().manual_seed(42)
            expected = [torch.randint(100, (3,), generator=gen) for _ in range(count)]
            local = []
            for index, starts in validation_batches(count, 3, 100, rank, world):
                assert index % world == rank
                assert torch.equal(starts, expected[index])
                local.append((index, float(starts.sum()), {"nll_sum": float(starts.sum()),
                              "nll_count": index + 1}, {"nll_sum": 1, "nll_count": 1}))
            loss, sums, occurrences = collect_validation(local, world)
            values = torch.stack(expected).sum(dim=1).float()
            assert loss.item() == values.mean().item()
            assert sums == {"nll_sum": values.sum().item(), "nll_count": sum(range(1, count + 1))}
            assert occurrences == {"nll_sum": count, "nll_count": count}
    finally:
        dist.destroy_process_group()


def test_four_rank_validation_matches_serial_with_uneven_and_empty_partitions(tmp_path):
    mp.spawn(check_ranks, args=((tmp_path / "init").as_uri(),), nprocs=4)
