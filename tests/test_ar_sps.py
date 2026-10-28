"""Offline correctness gates for pretrained AR-to-SPS adaptation."""
import copy
import json
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from modeling.models.sps.pretrained import PretrainedSPS
from scripts.ar_sps_train import optimizer_for
from scripts.ar_sps_common import loss
from scripts.ar_sps_run import calibrated_plan, reserve
from scripts import ar_sps_config as c


def model(sps=True):
    torch.manual_seed(13)
    config = LlamaConfig(vocab_size=31, hidden_size=24, intermediate_size=48,
                         num_hidden_layers=2, num_attention_heads=6, num_key_value_heads=2,
                         max_position_embeddings=128, tie_word_embeddings=True,
                         attention_dropout=0., rope_theta=100000.)
    config._attn_implementation = 'sdpa'
    return PretrainedSPS(LlamaForCausalLM(config), sps=sps).eval()


def test_native_parity_and_real_state_isolation():
    m = model(False)
    ids = torch.tensor([[2, 4, 8, 6]])
    expected = m.backbone(ids, output_hidden_states=True, use_cache=False)
    torch.testing.assert_close(m(ids).logits, expected.logits, rtol=0, atol=0)
    captured = []
    handle = m.backbone.model.norm.register_forward_hook(lambda module, args, result: captured.append(result))
    m.sps = True
    m(ids)
    handle.remove()
    torch.testing.assert_close(captured[0][:, ::2], expected.hidden_states[-1], atol=1e-6, rtol=1e-5)


def test_future_causality_and_prefix_equivalence():
    m = model()
    ids = torch.tensor([[2, 4, 8, 6]])
    before = m(ids).logits
    changed = ids.clone(); changed[:, 2:] = torch.tensor([9, 10])
    torch.testing.assert_close(m(changed).logits[:, :2], before[:, :2], atol=1e-6, rtol=1e-5)
    for length in range(1, 5):
        torch.testing.assert_close(m(ids[:, :length]).logits[:, -1], before[:, length-1], atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match='cached'):
        m(ids, past_key_values=object())


def test_freeze_then_unfreeze_ties_and_restore():
    m = model(); ids = torch.tensor([[2, 4, 8, 6]])
    initial = {k: v.clone() for k, v in m.backbone.state_dict().items()}
    prediction = m.prediction.clone()
    optimizer = optimizer_for(m, True)
    loss(m, ids).backward(); optimizer.step(); optimizer.zero_grad()
    assert not torch.equal(prediction, m.prediction)
    assert all(torch.equal(initial[k], v) for k, v in m.backbone.state_dict().items())
    optimizer = optimizer_for(m, False)
    params = [p for group in optimizer.param_groups for p in group['params']]
    assert len(params) == len({id(p) for p in params})
    assert m.backbone.lm_head.weight is m.backbone.model.embed_tokens.weight
    loss(m, ids).backward(); optimizer.step()
    assert any(not torch.equal(initial[k], v) for k, v in m.backbone.state_dict().items())
    restored = model(); restored.load_state_dict(copy.deepcopy(m.state_dict()))
    torch.testing.assert_close(restored(ids).logits, m(ids).logits, atol=0, rtol=0)
    assert restored.backbone.lm_head.weight is restored.backbone.model.embed_tokens.weight


def test_calibration_and_budget(tmp_path, monkeypatch):
    import scripts.ar_sps_run as run
    monkeypatch.setattr(run.e, 'ARTIFACTS_ROOT', tmp_path)
    probes = {'staged': {'embedding': 1., 'full': 2.}, 'direct': {'full': 2.}, 'ar': {'full': 1.}}
    plan = calibrated_plan(probes)
    assert c.MIN_STEPS <= plan['steps'] <= c.MAX_STEPS
    assert 0 < plan['stage_steps'] < plan['steps']
    with pytest.raises(RuntimeError):
        calibrated_plan({'bad': {'full': float('nan')}})
    reserve('prepare', 'data')
    for i in range(3): reserve('train', str(i))
    for i in range(5): reserve('eval', str(i))
    with pytest.raises(RuntimeError, match='cap'):
        reserve('train', 'retry')
    assert sum(a['gpu_hours'] for a in json.loads((tmp_path/'budget.json').read_text())['attempts']) <= c.GPU_HOURS


def test_content_split_and_paired_intervals():
    from scripts.ar_sps_data import split_for
    from scripts.ar_sps_report import paired_interval
    assert split_for('same document') == split_for('same document')
    assert paired_interval([0, 1], [0, 1]) == [0., 0.]
    with pytest.raises(ValueError): paired_interval([1], [])
