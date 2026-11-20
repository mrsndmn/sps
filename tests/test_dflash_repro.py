import torch
from torch import nn

from scripts import dflash_repro as jobs, dflash_repro_train as training


def test_epoch_shuffling_and_anchor_sampling():
    assert training.total_steps(93_216) == 11_652
    order = list(range(24))
    assert set(training.epoch_order(order, 0)) == set(order)
    assert training.epoch_order(order, 0) != training.epoch_order(order, 1)

    mask = [False] * 12 + [True] * 700
    first = training.sample_anchors(mask, 0, 37)
    assert first == training.sample_anchors(mask, 0, 37)
    assert first != training.sample_anchors(mask, 1, 37)
    assert len(set(first)) == training.MAX_ANCHORS
    assert all(i >= 12 and i + training.PROPOSALS < len(mask) for i in first)


def test_block_visibility_does_not_reveal_supervised_future_or_other_blocks():
    starts = torch.tensor([2, 5])
    ctx = torch.tensor(12)
    for q, correct in [(0, [0, 1, *range(12, 28)]), (31, [*range(5), *range(28, 44)])]:
        keys = training.block_visibility(starts, ctx, torch.tensor(q), torch.arange(44))
        assert keys.nonzero().flatten().tolist() == correct


def test_checkpoint_restores_optimizer_cursor_and_schedule(tmp_path):
    def create():
        model = nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
        return model, optimizer, schedule

    def update(model, optimizer, schedule):
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(2, 3)).square().sum().backward()
        optimizer.step()
        schedule.step()

    original = create()
    update(*original)
    folder = tmp_path / "step-00001"
    training.save_checkpoint(folder, *original, 1, "protocol")
    restored = create()
    assert training.load_checkpoint(folder, *restored, "protocol") == 1
    update(*original)
    update(*restored)
    for expected, actual in zip(original[0].parameters(), restored[0].parameters(), strict=True):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert original[2].state_dict() == restored[2].state_dict()


def test_worker_payload_requests_matching_training_topology_and_single_eval_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "signature", lambda: "abc" * 21)
    for phase in ("probe", "train"):
        body = jobs.payload(phase, None, tmp_path)
        assert body["instance_type"] == "a100.4gpu"
        assert "--nproc-per-node=4" in body["script"]
        assert body["processes_per_worker"] == 1
    body = jobs.payload("eval", 10000, tmp_path)
    assert body["instance_type"] == "a100.1gpu"
    assert "--phase eval --step 10000" in body["script"]
    assert "torch.distributed" not in body["script"]
