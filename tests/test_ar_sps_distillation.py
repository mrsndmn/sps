"""Distillation objective, frozen source teacher and immutable CE controls."""
import json
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from scripts import ar_sps_common as common
from scripts import ar_sps_config as c
from scripts import ar_sps_data as data
from scripts import ar_sps_run as run


def test_kl_direction_temperature_and_teacher_no_gradient():
    torch.manual_seed(4)
    logits = torch.randn(2, 3, 7, requires_grad=True)
    teacher = torch.randn(2, 3, 7, requires_grad=True)
    labels = torch.randint(7, (2, 3))
    objective, ce, kd = common.distillation_loss(logits, teacher, labels, .5, 2.)
    q = (teacher.detach()/2).softmax(-1)
    expected_kl = (q*(q.log()-(logits/2).log_softmax(-1))).sum(-1).mean()*4
    expected_ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    torch.testing.assert_close(kd, expected_kl)
    torch.testing.assert_close(objective, (expected_ce+expected_kl)/2)
    objective.backward()
    assert teacher.grad is None
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert not ce.requires_grad and not kd.requires_grad
    same, _, kl_same = common.distillation_loss(logits, logits.detach(), labels, 1., 1.)
    torch.testing.assert_close(kl_same, torch.zeros_like(kl_same), atol=1e-7, rtol=0)


def test_ce_unchanged_and_invalid_objectives():
    logits = torch.randn(2, 3, 7)
    labels = torch.randint(7, (2, 3))
    objective, _, kd = common.distillation_loss(logits, None, labels, 0., 1.)
    assert torch.equal(objective, F.cross_entropy(logits.flatten(0, 1), labels.flatten()))
    assert kd == 0
    for weight, temperature in [(-1., 1.), (2., 1.), (.5, 0.), (.5, float('nan'))]:
        with pytest.raises(ValueError): common.distillation_loss(logits, logits, labels, weight, temperature)
    with pytest.raises(ValueError): common.distillation_loss(logits, None, labels, .5, 1.)


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Embedding(11, 11)
        self.seen = None

    def forward(self, ids):
        self.seen = ids.detach().clone()
        return SimpleNamespace(logits=self.table(ids))


def test_frozen_teacher_is_independent_and_targets_are_next_token(monkeypatch):
    teacher = TinyLM()
    monkeypatch.setattr(common, 'load_model', lambda recipe, device: teacher)
    teacher = common.load_teacher(torch.device('cpu'))
    student = TinyLM()
    assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
    before = teacher.table.weight.detach().clone()
    batch = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
    optimizer = torch.optim.SGD(student.parameters(), lr=.1)
    objective, ce, kd = common.loss(student, batch, teacher=teacher, weight=.5, components=True)
    torch.testing.assert_close(teacher.seen, batch[:, :-1])
    torch.testing.assert_close(student.seen, batch[:, :-1])
    torch.testing.assert_close(ce, F.cross_entropy(student(batch[:, :-1]).logits.flatten(0, 1), batch[:, 1:].flatten()))
    objective.backward(); optimizer.step()
    assert teacher.table.weight.grad is None
    assert torch.equal(before, teacher.table.weight)
    assert teacher.table.weight.untyped_storage().data_ptr() != student.table.weight.untyped_storage().data_ptr()


def test_reference_data_preserves_order_and_rejects_tampering(tmp_path, monkeypatch):
    reference = tmp_path/'reference'
    reference.mkdir()
    arrays = {s: np.arange((c.CONTEXT+1)*2, dtype=np.uint32).reshape(2, -1) for s in ('train', 'val', 'test')}
    hashes = {}
    for split, array in arrays.items():
        path = reference/f'{split}.bin'; path.write_bytes(array.tobytes())
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(design_signature=c.REFERENCE_SIGNATURE, model=c.MODEL, revision=c.REVISION,
                    dataset=data.e.DATASET_REPO, dataset_revision=data.e.DATASET_REVISION,
                    row_length=c.CONTEXT+1, dtype='uint32', rows={s: 2 for s in arrays})
    (reference/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(data, 'reference_root', lambda: reference)
    monkeypatch.setattr(c, 'REFERENCE_DATA_SHA256', hashes)
    monkeypatch.setattr(c, 'DISTILLATION', True)
    monkeypatch.setattr(data.e, 'RUN_ROOT', tmp_path/'new')
    monkeypatch.setattr(data.e, 'design_signature', lambda: 'new-signature')
    data.prepare()
    np.testing.assert_array_equal(data.rows('train'), arrays['train'])
    assert json.loads((reference/'manifest.json').read_text()) == manifest
    damaged = arrays['train'].copy(); damaged[0, 0] += 1
    (reference/'train.bin').write_bytes(damaged.tobytes())
    with pytest.raises(RuntimeError, match='checksum'):
        data.prepare()


def test_matched_budget_must_not_shrink(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'DISTILLATION', True)
    monkeypatch.setattr(c, 'TRAIN_HOURS', 2)
    monkeypatch.setattr(run.e, 'COMMON_ROOT', tmp_path)
    root = tmp_path/c.REFERENCE_SUITE; root.mkdir()
    parent = dict(design_signature=c.REFERENCE_SIGNATURE, steps=c.MAX_STEPS,
                  stage_steps=int(c.MAX_STEPS*c.STAGE_FRACTION), text_tokens=c.MAX_STEPS*c.GLOBAL_BATCH*c.CONTEXT)
    (root/'prepared.json').write_text(json.dumps(parent))
    result = run.calibrated_plan({'staged_kd': {'full': .6, 'embedding': .4}})
    assert result['steps'] == parent['steps'] and result['stage_steps'] == parent['stage_steps']
    with pytest.raises(RuntimeError, match='matched exposure'):
        run.calibrated_plan({'staged_kd': {'full': 10., 'embedding': 10.}})


def test_external_control_signature_is_explicit(tmp_path, monkeypatch):
    from scripts import ar_sps_report as report
    monkeypatch.setattr(report.e, 'design_signature', lambda: 'current')
    monkeypatch.setattr(c, 'PRIMARY_METRICS', {})
    baseline, candidate = tmp_path/'control', tmp_path/'final'
    for path, signature in ((baseline, c.REFERENCE_SIGNATURE), (candidate, 'current')):
        path.mkdir()
        (path/'complete.json').write_text(json.dumps(dict(design_signature=signature)))
        (path/'heldout.json').write_text(json.dumps(dict(step=10, nll=1., per_sequence=[1., 1.])))
    with pytest.raises(RuntimeError, match='Stale'):
        report.compare(baseline, candidate)
    result = report.compare(baseline, candidate, baseline_signature=c.REFERENCE_SIGNATURE, matched_exposure=True)
    assert result['matched_exposure'] and result['observed_retention']
    (candidate/'complete.json').write_text(json.dumps(dict(design_signature=c.REFERENCE_SIGNATURE)))
    with pytest.raises(RuntimeError, match='Stale'):
        report.compare(baseline, candidate, baseline_signature=c.REFERENCE_SIGNATURE)


def test_distillation_report_uses_only_declared_controls(tmp_path, monkeypatch):
    from scripts import ar_sps_report as report
    from types import SimpleNamespace
    monkeypatch.setattr(c, 'REFERENCE_CONTROLS', ('smollm2_staged_sps',))
    monkeypatch.setattr(report.e, 'COMMON_ROOT', tmp_path)
    monkeypatch.setattr(data, 'rows', lambda _: None)
    monkeypatch.setattr(report.e, 'collect_experiments', lambda: [SimpleNamespace(name='kd')])
    monkeypatch.setattr(report.e, 'output_dir', lambda _: tmp_path/'kd')
    for label in ('stage_endpoint','final'):
        p=tmp_path/'kd/eval'/label;p.mkdir(parents=True);(p/'complete.json').write_text('{}')
    calls=[]
    def compare(baseline,candidate,**kwargs):
        calls.append((baseline,candidate,kwargs));return {}
    monkeypatch.setattr(report,'compare',compare)
    result=report.distillation_report()
    assert len(result['against_original'])==2
    assert len(result['matched_endpoint_controls'])==2
    assert all('smollm2_staged_sps' in str(b) for b,_,kw in calls if kw.get('matched_exposure'))
    assert all(kw['baseline_signature']==c.REFERENCE_SIGNATURE for _,_,kw in calls)
