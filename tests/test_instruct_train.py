"""Masked assistant training must preserve target alignment and token weighting."""
import copy

import pytest
import torch
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from modeling.models.sps.pretrained import PretrainedSPS
from scripts.instruct_common import masked_loss, loss, nll


def tiny(family, sps=True):
    cfg, cls = (LlamaConfig, LlamaForCausalLM) if family == 'llama' else (Qwen3Config, Qwen3ForCausalLM)
    config = cfg(vocab_size=31, hidden_size=24, intermediate_size=48, num_hidden_layers=2,
                 num_attention_heads=6, num_key_value_heads=2, head_dim=8,
                 max_position_embeddings=128, tie_word_embeddings=True, attention_dropout=0.)
    config._attn_implementation = 'sdpa'
    return PretrainedSPS(cls(config), sps=sps).eval()


def test_chunked_kd_value_and_gradient_match_dense_masked_reference():
    torch.manual_seed(8)
    logits = torch.randn(2, 6, 13, requires_grad=True)
    teacher = torch.randn_like(logits, requires_grad=True)
    targets = torch.randint(0, 13, (2, 6))
    mask = torch.tensor([[0, 1, 1, 0, 0, 1], [1, 0, 0, 0, 1, 1]], dtype=torch.bool)
    objective, ce, kd, count = masked_loss(logits, teacher, targets, mask, weight=.5, temperature=2., chunk_size=2)
    selected = logits[mask]
    ce_ref = F.cross_entropy(selected, targets[mask], reduction='sum')
    kd_ref = F.kl_div(F.log_softmax(selected/2., dim=-1),
                     F.log_softmax(teacher.detach()[mask]/2., dim=-1),
                     log_target=True, reduction='sum')*4
    expected = .5*ce_ref+.5*kd_ref
    torch.testing.assert_close(objective, expected)
    torch.testing.assert_close(ce, ce_ref)
    torch.testing.assert_close(kd, kd_ref)
    got_grad, = torch.autograd.grad(objective, logits, retain_graph=True)
    ref_grad, = torch.autograd.grad(expected, logits)
    torch.testing.assert_close(got_grad, ref_grad)
    assert count == mask.sum().item()
    assert torch.count_nonzero(got_grad[~mask]) == 0
    assert teacher.grad is None


def test_ddp_accumulation_uses_global_token_denominator():
    # Simulate DDP's final average across ranks with unequal supervised counts.
    parameter = torch.tensor(.3, requires_grad=True)
    rows = [(0, [0, 1, 1]), (0, [1]), (1, [0]), (1, [1, 0])]
    count = sum(len(labels) for _, labels in rows)
    rank_gradients = []
    for rank in (0, 1):
        local_sum = 0.
        for r, labels in rows:
            if r == rank:
                logits = torch.stack((parameter, -parameter)).expand(len(labels), 2)
                local_sum = local_sum + F.cross_entropy(logits, torch.tensor(labels), reduction='sum')
        rank_gradients.append(torch.autograd.grad(local_sum * 2/count, parameter, retain_graph=True)[0])
    all_labels = torch.tensor([label for _, labels in rows for label in labels])
    ref = F.cross_entropy(torch.stack((parameter, -parameter)).expand(count, 2), all_labels)
    reference_grad, = torch.autograd.grad(ref, parameter)
    torch.testing.assert_close(sum(rank_gradients)/2, reference_grad)


@pytest.mark.parametrize('family', ['llama', 'qwen3'])
def test_target_shift_frozen_prediction_and_teacher(family):
    torch.manual_seed(3)
    model = tiny(family)
    teacher = copy.deepcopy(model)
    teacher.sps = False
    teacher.requires_grad_(False)
    model.set_stage(True)
    row = dict(input_ids=[2, 4, 5, 6, 7], loss_mask=[0, 0, 0, 1, 1], mode='thinking', source_id='test')
    objective, _, _, count = loss(model, row, torch.device('cpu'), teacher=teacher, weight=.5)
    assert count == 2
    objective.backward()
    assert model.prediction.grad is not None and model.prediction.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    model.zero_grad(set_to_none=True)
    model.set_stage(False)
    value, ce, _, _ = loss(model, row, torch.device('cpu'))
    ids = torch.tensor([row['input_ids']])
    expected = F.cross_entropy(model(ids[:, :-1]).logits[:, 2:].float().flatten(0, 1), ids[:, 3:].flatten(), reduction='sum')
    torch.testing.assert_close(value, expected)
    value.backward()
    assert model.backbone.get_input_embeddings().weight.grad is not None
    # Instruct wrapper has not changed original AR's numerical path.
    model.sps = False
    torch.testing.assert_close(model(ids).logits, model.backbone(ids, use_cache=False).logits, rtol=0, atol=0)


@pytest.mark.parametrize('family', ['llama', 'qwen3'])
def test_nll_token_weighted_and_modes(family):
    model = tiny(family)
    data = [dict(input_ids=[2, 3, 4, 5], loss_mask=[0, 0, 1, 1], mode='thinking', source_id='a'),
            dict(input_ids=[2, 5, 6], loss_mask=[0, 0, 1], mode='non_thinking', source_id='b')]
    model.train()
    result = nll(model, data, torch.device('cpu'))
    assert model.training
    assert result['tokens'] == 3
    assert set(result['by_mode']) == {'thinking', 'non_thinking'}
    assert result['nll'] == pytest.approx(sum(r['nll']*r['tokens'] for r in result['per_example'])/3)


def test_empty_or_invalid_mask_fails():
    logits = torch.randn(1, 3, 7)
    targets = torch.tensor([[1, 2, 3]])
    with pytest.raises(ValueError, match='No supervised'):
        masked_loss(logits, None, targets, torch.zeros_like(targets))
    with pytest.raises(ValueError, match='binary'):
        masked_loss(logits, None, targets, torch.full_like(targets, 2))


def test_checkpoint_provenance_and_stage_metadata(tmp_path, monkeypatch):
    import experiments as e
    from types import SimpleNamespace
    from scripts import instruct_common as common
    monkeypatch.setattr(common, 'configuration', lambda: SimpleNamespace(MODEL='test/model', REVISION='pinned'))
    monkeypatch.setattr(e, 'design_signature', lambda: 'signature')
    model = tiny('qwen3')
    model.set_stage(True)
    optimizer = torch.optim.AdamW([model.prediction], lr=.01)
    plan = dict(steps=8, stage_steps=2, design_signature='signature')
    path = tmp_path/'checkpoint.pt'
    common.save(path, model, optimizer, 2, 3.5, True, plan)
    restored = tiny('qwen3')
    state = common.restore(path, restored, torch.device('cpu'))
    assert state['plan'] == plan and state['stage'] is True and state['step'] == 2
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)
    monkeypatch.setattr(e, 'design_signature', lambda: 'different')
    with pytest.raises(RuntimeError, match='provenance'):
        common.restore(path, restored, torch.device('cpu'))


@pytest.mark.parametrize('family', ['llama', 'qwen3'])
def test_optimizer_transition_unfreezes_backbone_and_checkpointed_gradients(family):
    from scripts.instruct_train import optimizer_for
    model = tiny(family)
    model.train()
    initial = {name: value.clone() for name, value in model.backbone.state_dict().items()}
    row = dict(input_ids=[2, 3, 4, 5], loss_mask=[0, 0, 1, 1], mode='thinking', source_id='phase')
    frozen_optimizer = optimizer_for(model, True)
    value, _, _, _ = loss(model, row, torch.device('cpu'))
    value.backward(); frozen_optimizer.step(); frozen_optimizer.zero_grad(set_to_none=True)
    assert len(frozen_optimizer.param_groups[0]['params']) == 1
    for name, value in model.backbone.state_dict().items():
        torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
    full_optimizer = optimizer_for(model, False)
    assert model.backbone.is_gradient_checkpointing
    value, _, _, _ = loss(model, row, torch.device('cpu'))
    value.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    full_optimizer.step()
    assert any(not torch.equal(value, initial[name]) for name, value in model.backbone.state_dict().items())
