"""Continuation provenance and complete draft checkpoint restoration."""
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

import experiments as e
from scripts import mtp_common as common
from modeling.models.sps.core import SPSConfig, SPSModelBase
from modeling.models.sps.mtp import SPSMTP


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def parent(tmp_path, monkeypatch):
    source = tmp_path/'source.pt'
    torch.save(dict(model_args=dict(hidden_size=16)), source)
    data = tmp_path/'data'
    data.mkdir()
    (data/'manifest.json').write_text('{}')
    root = tmp_path/'parents'
    directory = root/'s_sps_mtp_n3_emb_1b'
    directory.mkdir(parents=True)
    path = directory/'adaptation_final.pt'
    state = dict(horizons=3, source=str(source), new_embeddings=torch.ones(2, 16),
                 step=8, tokens=128, design_signature='parent-design')
    torch.save(state, path)
    marker = dict(checkpoint=str(path), design_signature='parent-design',
                  source_sha256=digest(source), data_manifest_sha256=digest(data/'manifest.json'))
    (directory/'trained.json').write_text(json.dumps(marker))
    monkeypatch.setattr(common, 'MTP_SOURCE', source)
    monkeypatch.setattr(common, 'MTP_DATA', data)
    monkeypatch.setattr(e, 'MTP_PARENT_ROOT', root, raising=False)
    monkeypatch.setattr(e, 'MTP_PARENT_INPUT_TOKENS', 128, raising=False)
    monkeypatch.setattr(e, 'MTP_PARENT_CONTEXT', 4, raising=False)
    monkeypatch.setattr(e, 'MTP_PARENT_STEP_TOKENS', 16, raising=False)
    return SimpleNamespace(path=path, state=state, marker=marker, directory=directory)


def test_parent_metadata_uses_parent_design_and_exact_sampler_position(parent):
    metadata = common.warmstart_metadata(3)
    assert metadata['path'] == str(parent.path)
    assert metadata['sha256'] == digest(parent.path)
    assert metadata['parent_tokens'] == 128
    assert metadata['parent_step'] == 8
    assert metadata['parent_samples_per_rank'] == 8
    assert metadata['parent_design_signature'] == 'parent-design'


@pytest.mark.parametrize('field,value', [('tokens', 127), ('horizons', 2),
    ('design_signature', 'different'), ('source', 'wrong')])
def test_parent_rejects_changed_checkpoint_identity(parent, field, value):
    parent.state[field] = value
    torch.save(parent.state, parent.path)
    with pytest.raises(RuntimeError):
        common.warmstart_metadata(3)


def test_parent_rejects_changed_data(parent):
    (common.MTP_DATA/'manifest.json').write_text('{"changed": true}')
    with pytest.raises(RuntimeError, match='manifest'):
        common.warmstart_metadata(3)


def tiny():
    return SPSMTP(SPSModelBase(SPSConfig(vocab_size=19, block_size=16, n_layer=1,
        n_head=2, hidden_size=16, intermediate_size=24, eos_token_id=16,
        pad_token_id=17, predict_token_id=18, window_size=0, dropout=0.0)), 3, backend='sdpa')


def test_restore_draft_roundtrips_all_weights_and_preserves_tying():
    original = tiny()
    state = dict(source=str(common.MTP_SOURCE), horizons=3, train_backbone=True,
                 new_embeddings=torch.randn_like(original.new_embeddings),
                 backbone={k: torch.randn_like(v) for k,v in original.reference.state_dict().items()})
    # Tied entries must agree; reject ambiguity rather than last-copy-wins.
    state['backbone']['lm_head.weight'] = state['backbone']['transformer.wte.weight']
    common.restore_draft(original, state)
    assert original.reference.transformer.wte.weight is original.reference.lm_head.weight
    for key,value in original.reference.state_dict().items():
        torch.testing.assert_close(value, state['backbone'][key])
    torch.testing.assert_close(original.new_embeddings, state['new_embeddings'])


def test_restore_rejects_missing_backbone_and_disagreeing_tied_weights():
    mtp = tiny()
    state = dict(source=str(common.MTP_SOURCE), horizons=3, train_backbone=True,
                 new_embeddings=mtp.new_embeddings.detach().clone())
    with pytest.raises(RuntimeError, match='backbone'):
        common.restore_draft(mtp, state)
    state['backbone'] = {k:v.clone() for k,v in mtp.reference.state_dict().items()}
    state['backbone']['lm_head.weight'].add_(1)
    with pytest.raises(RuntimeError, match='tied'):
        common.restore_draft(mtp, state)


def test_parent_rejects_wrong_embedding_width(parent):
    parent.state['new_embeddings'] = torch.ones(2, 7)
    torch.save(parent.state, parent.path)
    with pytest.raises(RuntimeError, match='shape'):
        common.warmstart_metadata(3)


def test_new_suite_default_loads_parent_and_explicit_checkpoint_restores_full(parent, monkeypatch):
    monkeypatch.setattr(e, 'MTP_UNFREEZE', True, raising=False)
    monkeypatch.setattr(common, 'load_reference', lambda device: tiny().reference.to(device))
    draft = common.load_mtp(3, 'cpu', backend='sdpa', train_backbone=True)
    torch.testing.assert_close(draft.new_embeddings, parent.state['new_embeddings'])
    assert draft.train_backbone
    assert draft.reference.transformer.wte.weight.requires_grad
    with torch.no_grad():
        draft.reference.transformer.wte.weight.add_(.5)
    path = parent.directory/'full.pt'
    torch.save(dict(horizons=3, source=str(common.MTP_SOURCE), train_backbone=True,
        backbone=draft.reference.state_dict(), new_embeddings=draft.new_embeddings.detach()), path)
    loaded = common.load_mtp(3, 'cpu', checkpoint_path=path, backend='sdpa')
    for key, value in draft.reference.state_dict().items():
        torch.testing.assert_close(loaded.reference.state_dict()[key], value)
    assert loaded.reference.transformer.wte.weight is loaded.reference.lm_head.weight


def test_existing_suite_keeps_fresh_embeddings_and_frozen_backbone(parent, monkeypatch):
    monkeypatch.setattr(e, 'MTP_UNFREEZE', False, raising=False)
    monkeypatch.setattr(common, 'load_reference', lambda device: tiny().reference.to(device))
    monkeypatch.setattr(common, 'warmstart_metadata', lambda *a: pytest.fail('Old suite must not warmstart'))
    draft = common.load_mtp(3, 'cpu', backend='sdpa')
    expected = draft.reference.transformer.wte.weight[draft.config.predict_token_id].repeat(2, 1)
    torch.testing.assert_close(draft.new_embeddings, expected)
    assert not any(p.requires_grad for p in draft.reference.parameters())
