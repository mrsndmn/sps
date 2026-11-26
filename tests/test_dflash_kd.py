from dataclasses import replace
import torch
from torch import nn

from experiments.dflash_repro_train import HardDFlash
from experiments.dflash_kd_train import SoftKD

from workloads.distillation import (
    dflash_kd as jobs,
    dflash_kd_train as training,
    dflash_repro_train as hard,
)

(BASELINE,) = HardDFlash.experiment_configs()
KD = next(x for x in SoftKD.experiment_configs() if x.name == "qwen3-4b-dflash-b16-soft-kd")


def test_teacher_indices_use_previous_token_and_reuse_overlapping_positions():
    weights = torch.zeros(2, BASELINE.proposals)
    weights[0, [0, 1, 4]] = 1
    weights[1, [0, 2]] = 1
    kept, unique, inverse = training.teacher_indices(
        [4, 5], weights, weights.device, KD
    )
    assert kept.tolist() == [0, 1, 4, 15, 17]
    assert unique.tolist() == [4, 5, 7, 8]
    assert unique[inverse].tolist() == [4, 5, 8, 5, 7]


def test_chunked_kl_matches_direct_kl_and_freezes_head(monkeypatch):
    class DummyDraft(nn.Module):
        def __init__(self):
            super().__init__()
            self.states = nn.Parameter(torch.randn(1, BASELINE.width * 2, 3))

        def forward(self, **kwargs):
            return self.states

    draft = DummyDraft()
    head = nn.Linear(3, 4, bias=False)
    objective = training.SoftBlockLoss(
        draft, head, experiment=replace(KD, head_chunk=2)
    )
    weights = torch.zeros(2, BASELINE.proposals)
    weights[0, [0, 1, 4]] = torch.tensor([1, 0.9, 0.5])
    weights[1, [0, 2]] = torch.tensor([1, 0.7])
    kept, unique, inverse = training.teacher_indices(
        [4, 5], weights, weights.device, KD
    )
    logp = torch.randn(len(unique), 4).log_softmax(-1)
    actual = objective(None, None, None, None, weights, kept, logp, inverse)

    states = draft.states.view(2, BASELINE.width, 3)[:, 1:].reshape(-1, 3)[kept]
    logq = (states @ head.weight.detach().T).log_softmax(-1)
    expected = (
        ((logp[inverse] - logq) * logp[inverse].exp()).sum(-1) * weights.flatten()[kept]
    ).sum() / weights.sum()
    torch.testing.assert_close(actual, expected)
    (expected_grad,) = torch.autograd.grad(expected, draft.states)
    actual.backward()
    torch.testing.assert_close(draft.states.grad, expected_grad)
    assert head.weight.grad is None


def test_payload_keeps_global_batch_and_eval_gpu_count(tmp_path, monkeypatch):
    assert KD.global_batch == BASELINE.global_batch == 24
    assert hard.total_steps(93216, KD) == 11652
    for phase in ("probe", "train"):
        body = jobs.runner.payload(
            KD,
            phase,
            None,
            tmp_path,
            "scripts/distillation/dflash_kd.py",
            jobs.runner.worker_arguments(training, KD, phase, None),
        )
        assert body["instance_type"] == "a100.8gpu"
        assert "--nproc-per-node=8" in body["script"]
    assert (
        jobs.runner.payload(
            KD,
            "eval",
            5000,
            tmp_path,
            "scripts/distillation/dflash_kd.py",
            jobs.runner.worker_arguments(training, KD, "eval", 5000),
        )["instance_type"]
        == "a100.1gpu"
    )


def test_group_factory_returns_independent_configs():
    first = SoftKD.experiment_configs()
    second = SoftKD.experiment_configs()
    first[0].milestones.clear()
    assert second[0].milestones == [5000, 10000]
    assert first is not second
