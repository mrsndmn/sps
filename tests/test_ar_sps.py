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


def test_report_exposure_and_per_task_retention(tmp_path, monkeypatch):
    from scripts import ar_sps_report as report
    monkeypatch.setattr(report.c, 'PRIMARY_METRICS', {'task_a': 'acc,none', 'task_b': 'acc,none'})
    monkeypatch.setattr(report.e, 'design_signature', lambda: 'test')
    def write(root, step, scores):
        root.mkdir(parents=True, exist_ok=True)
        (root/'complete.json').write_text(json.dumps({'design_signature': 'test'}))
        (root/'heldout.json').write_text(json.dumps({'step': step, 'nll': 1., 'per_sequence': [1., 1.]}))
        for task, score in zip(('task_a', 'task_b'), scores):
            (root/f'{task}.json').write_text(json.dumps({
                'results': {task: {'acc,none': score}},
                'samples': {task: [{'doc_id': 0, 'doc_hash': 'same', 'acc': score}]}}))
    base = tmp_path/'control/final'; candidate = tmp_path/'staged/final'
    write(base, 10, [.5, .5]); write(candidate, 10, [.4, .6])
    result = report.compare(base, candidate, matched_exposure=True)
    assert not result['observed_retention']  # equal mean cannot hide a task drop
    assert result['matched_exposure']
    assert result['candidate_text_tokens'] == 10*c.GLOBAL_BATCH*c.CONTEXT
    write(candidate, 9, [.5, .5])
    with pytest.raises(RuntimeError, match='Unequal exposure'):
        report.compare(base, candidate, matched_exposure=True)
    result = report.compare(base, candidate)
    assert not result['matched_exposure']
    assert result['candidate_step'] == 9


def test_audit_checks_reference_and_production_causality():
    from scripts.ar_sps_evaluate import audit_model
    result = audit_model(model(False), torch.tensor([[2, 4, 8, 6]]), 2)
    assert result['native_max_abs'] == 0
    assert result['float32']['causal_max_abs'] == 0
    assert result['bfloat16']['causal_max_abs'] == 0
    assert result['float64_reference']['cross_length_max_abs'] < c.PARITY_ATOL


def test_audit_rejects_future_leakage(monkeypatch):
    from scripts.ar_sps_evaluate import audit_model
    m = model(False)
    original = m.forward
    def leaky(ids, **kwargs):
        output = original(ids, **kwargs)
        if m.sps:
            output.logits = output.logits + ids[:, -1, None, None].float()
        return output
    monkeypatch.setattr(m, 'forward', leaky)
    with pytest.raises(AssertionError):
        audit_model(m, torch.tensor([[2, 4, 8, 6]]), 2)


def test_audit_records_shape_drift_without_relaxing_reference(monkeypatch):
    from scripts.ar_sps_evaluate import audit_model
    m = model(False)
    original = m.forward
    def shape_dependent(ids, **kwargs):
        output = original(ids, **kwargs)
        if m.sps and ids.shape[1] == 2 and m.prediction.dtype == torch.float32:
            output.logits = output.logits.float() + .001
        return output
    monkeypatch.setattr(m, 'forward', shape_dependent)
    result = audit_model(m, torch.tensor([[2, 4, 8, 6]]), 2)
    assert result['float32']['cross_length_max_abs'] > c.PARITY_ATOL
    assert result['float32']['causal_max_abs'] == 0
    assert result['float64_reference']['cross_length_max_abs'] < c.PARITY_ATOL
