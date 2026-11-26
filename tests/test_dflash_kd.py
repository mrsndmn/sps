from dataclasses import replace
import torch
from torch import nn

from workloads.distillation import dflash_kd as jobs, dflash_kd_train as training, dflash_repro_train as hard


def test_teacher_indices_use_previous_token_and_reuse_overlapping_positions():
    weights = torch.zeros(2, hard.EXPERIMENT.proposals)
    weights[0, [0, 1, 4]] = 1
    weights[1, [0, 2]] = 1
    kept, unique, inverse = training.teacher_indices([4, 5], weights, weights.device)
    assert kept.tolist() == [0, 1, 4, 15, 17]
    assert unique.tolist() == [4, 5, 7, 8]
    assert unique[inverse].tolist() == [4, 5, 8, 5, 7]


def test_chunked_kl_matches_direct_kl_and_freezes_head(monkeypatch):
    class DummyDraft(nn.Module):
        def __init__(self):
            super().__init__()
            self.states = nn.Parameter(torch.randn(1, hard.EXPERIMENT.width * 2, 3))

        def forward(self, **kwargs):
            return self.states

    draft = DummyDraft()
    head = nn.Linear(3, 4, bias=False)
    objective = training.SoftBlockLoss(draft, head, experiment=replace(training.EXPERIMENT, head_chunk=2))
    weights = torch.zeros(2, hard.EXPERIMENT.proposals)
    weights[0, [0, 1, 4]] = torch.tensor([1, .9, .5])
    weights[1, [0, 2]] = torch.tensor([1, .7])
    kept, unique, inverse = training.teacher_indices([4, 5], weights, weights.device)
    logp = torch.randn(len(unique), 4).log_softmax(-1)
    actual = objective(None, None, None, None, weights, kept, logp, inverse)

    states = draft.states.view(2, hard.EXPERIMENT.width, 3)[:, 1:].reshape(-1, 3)[kept]
    logq = (states @ head.weight.detach().T).log_softmax(-1)
    expected = (((logp[inverse] - logq) * logp[inverse].exp()).sum(-1)
                * weights.flatten()[kept]).sum() / weights.sum()
    torch.testing.assert_close(actual, expected)
    expected_grad, = torch.autograd.grad(expected, draft.states)
    actual.backward()
    torch.testing.assert_close(draft.states.grad, expected_grad)
    assert head.weight.grad is None


def test_payload_keeps_global_batch_and_eval_gpu_count(tmp_path, monkeypatch):
    assert training.EXPERIMENT.global_batch == hard.EXPERIMENT.global_batch == 24
    assert hard.total_steps(93216, training.EXPERIMENT) == 11652
    for phase in ("probe", "train"):
        body = jobs.runner.payload(training.EXPERIMENT, phase, None, tmp_path, "scripts/distillation/dflash_kd.py")
        assert body["instance_type"] == "a100.8gpu"
        assert "--nproc-per-node=8" in body["script"]
    assert jobs.runner.payload(training.EXPERIMENT, "eval", 5000, tmp_path, "scripts/distillation/dflash_kd.py")["instance_type"] == "a100.1gpu"
