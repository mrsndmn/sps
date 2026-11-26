import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from workloads.adaptation import qwen_adapter_run as cfg


def test_attention_mlp_scope():
    command = """from pathlib import Path
from workloads.adaptation import qwen_adapter_run as cfg
arm, = cfg.experiment_configs
assert arm.dora and not arm.rs and arm.lr == 1e-4
assert set(cfg.TARGETS) == {'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}
assert cfg.ROOT.name == 'qwen3-base-adapters-attention-mlp'
for phase, count in [('train',8),('eval',1)]:
    body = cfg.payload(phase,arm,Path('/stage'))
    assert body['instance_type'] == f'a100.{count}gpu'
    assert body['env_variables']['SPS_ADAPTER_SCOPE'] == 'attention-mlp'
    assert ('--nproc_per_node=8' in body['script']) == (phase == 'train')
"""
    subprocess.run([sys.executable, '-c', command], check=True,
                   env={**os.environ, 'SPS_ADAPTER_SCOPE': 'attention-mlp'})


@pytest.fixture
def select_scope(monkeypatch):
    try:
        with monkeypatch.context() as env:
            def select(scope):
                env.setenv('SPS_ADAPTER_SCOPE', scope)
                importlib.reload(cfg)
            yield select
    finally:
        importlib.reload(cfg)


def test_grid_and_resources(select_scope):
    select_scope('attention')
    grid = cfg.experiment_configs
    assert len({(arm.dora, arm.rs, arm.lr) for arm in grid}) == len(grid) == 8
    for phase, gpus in [('train', 4), ('eval', 1)]:
        body = cfg.payload(phase, grid[0], Path('/tmp/staged'))
        assert body['instance_type'] == f'a100.{gpus}gpu'
        assert ('--nproc_per_node=4' in body['script']) == (phase == 'train')


def test_winner_requires_all_finals(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, 'ROOT', tmp_path)
    for index, arm in enumerate(cfg.experiment_configs):
        root = cfg.directory(arm)
        root.mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            cfg.winner('commit')
        (root/'final.pt').write_bytes(arm.name.encode())
        (root/'trained.json').write_text(json.dumps(dict(arm=arm.name, step=cfg.END,
            commit='commit', val_nll=10-index, sha256=cfg.digest(root/'final.pt'))))
    assert cfg.winner('commit') == cfg.experiment_configs[-1]
    with pytest.raises(ValueError):
        cfg.winner('different')
    (root/'final.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        cfg.winner('commit')


@pytest.mark.skipif(not cfg.WHEEL.exists(), reason='requires the project pinned PEFT wheel')
@pytest.mark.parametrize('scope,index', [('attention', index) for index in range(4)] + [('attention-mlp', 0)])
def test_adapter_roundtrip(scope, index, select_scope):
    select_scope(scope)
    arm = cfg.experiment_configs[index]
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from modeling.models.sps.pretrained import PretrainedSPS
    from workloads.adaptation.qwen_adapter_worker import attach
    import copy
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    model = PretrainedSPS(Qwen3ForCausalLM(config)).eval()
    other = copy.deepcopy(model)
    ids = torch.tensor([[1, 2, 3, 4]])
    before = model(ids).logits.detach()
    params = attach(model, arm)
    torch.testing.assert_close(model(ids).logits, before)
    model(ids).logits.square().mean().backward()
    assert all(param.grad is not None for param in [*params, model.prediction])
    torch.optim.AdamW([*params, model.prediction], lr=.001).step()
    attach(other, arm)
    other.load_state_dict({name: p for name, p in model.named_parameters() if p.requires_grad}, strict=False)
    torch.testing.assert_close(model(ids).logits, other(ids).logits)
    assert not torch.equal(before, other(ids).logits)


def test_archived_winner_uses_the_checkpoint_cohort(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    for i, arm in enumerate(cfg.experiment_configs):
        folder = cfg.directory(arm)
        folder.mkdir(parents=True)
        weights = folder / "final.pt"
        weights.write_bytes(arm.name.encode())
        (folder / "trained.json").write_text(json.dumps(dict(
            arm=arm.name, step=cfg.END, commit="archived", val_nll=i,
            sha256=cfg.digest(weights))))
    assert cfg.winner() == cfg.experiment_configs[0]
    marker = folder / "trained.json"
    row = json.loads(marker.read_text())
    row["commit"] = "another cohort"
    marker.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        cfg.winner()
