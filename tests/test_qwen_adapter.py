import json
from pathlib import Path

import pytest

from scripts import qwen_adapter_run as cfg


def test_grid_and_resources():
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
        (root/'trained.json').write_text(json.dumps(dict(arm=arm.name, step=cfg.END,
            commit='commit', val_nll=10-index)))
        (root/'final.pt').touch()
    assert cfg.winner('commit') == cfg.experiment_configs[-1]
    with pytest.raises(RuntimeError):
        cfg.winner('different')


@pytest.mark.skipif(not cfg.WHEEL.exists(), reason='requires the project pinned PEFT wheel')
@pytest.mark.parametrize('arm', cfg.experiment_configs[:4], ids=lambda arm: arm.name)
def test_adapter_roundtrip(arm):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from modeling.models.sps.pretrained import PretrainedSPS
    from scripts.qwen_adapter_worker import attach
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
