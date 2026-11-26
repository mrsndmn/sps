import pytest
import torch
from torch import nn

from workloads.distillation import dflash_repro as jobs, dflash_repro_train as training


def test_epoch_shuffling_and_anchor_sampling():
    assert training.total_steps(93_216) == 11_652
    order = list(range(24))
    assert set(training.epoch_order(order, 0)) == set(order)
    assert training.epoch_order(order, 0) != training.epoch_order(order, 1)

    mask = [False] * 12 + [True] * 700
    first = training.sample_anchors(mask, 0, 37)
    assert first == training.sample_anchors(mask, 0, 37)
    assert first != training.sample_anchors(mask, 1, 37)
    assert len(set(first)) == training.EXPERIMENT.max_anchors
    assert all(i >= 12 and i + training.EXPERIMENT.proposals < len(mask) for i in first)


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
    for phase in ("probe", "train"):
        body = jobs.payload(training.EXPERIMENT, phase, None, tmp_path, "scripts/distillation/dflash_repro.py")
        assert body["instance_type"] == "a100.4gpu"
        assert "--nproc-per-node=4" in body["script"]
        assert body["processes_per_worker"] == 1
    body = jobs.payload(training.EXPERIMENT, "eval", 10000, tmp_path, "scripts/distillation/dflash_repro.py")
    assert body["instance_type"] == "a100.1gpu"
    assert "--step 10000" in body["script"]
    assert "torch.distributed" not in body["script"]


def test_milestone_evaluation_waits_for_published_training(tmp_path, monkeypatch):
    from dataclasses import replace
    import pytest
    from orchestration.named import publication

    exp = replace(training.EXPERIMENT, output_root=tmp_path / "experiment")
    monkeypatch.setattr(training, "EXPERIMENT", exp)
    monkeypatch.setattr(publication, "COMMON_ROOT", tmp_path)
    weights = training.checkpoint_path(5000) / "weights.safetensors"
    weights.parent.mkdir(parents=True)
    weights.touch()
    calls = []
    monkeypatch.setattr(jobs, "submit_job", lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match="Complete DFlash training"):
        jobs.submit_phase(exp, training, "eval", 5000, {}, "test")
    jobs.marker(exp, "train").touch()
    jobs.submit_phase(exp, training, "eval", 5000, {}, "test")
    assert len(calls) == 1
    exp.output("eval", 5000).mkdir()
    jobs.submit_phase(exp, training, "eval", 5000, {}, "test")
    assert len(calls) == 1


def test_checkpoint_inputs_detect_reordered_selection(tmp_path, monkeypatch):
    from workloads.distillation import dflash_repro_eval as evaluation

    selection = tmp_path / "selection.json"
    monkeypatch.setattr(training.previous, "selection_path", lambda: selection)
    selection.write_text('{"train":[1,2],"val":[3]}')
    inputs = training.training_inputs(training.EXPERIMENT)
    training.launch.store_json(tmp_path / "inputs.json", inputs)
    selection.write_text('{"train":[2,1],"val":[3]}')
    with pytest.raises(ValueError, match="checkpoint training inputs"):
        evaluation.evaluate(tmp_path, tmp_path / "eval.json")
