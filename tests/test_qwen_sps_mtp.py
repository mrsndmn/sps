"""Native Qwen horizon causality and shared first-slot verification contract."""

import importlib.util
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from modeling.models.sps.pretrained import PretrainedSPS
from scripts.vllm_sps_model import QwenSPSMTP, horizon_targets


def tiny(head_dim=8):
    torch.manual_seed(7)
    cfg = Qwen3Config(
        vocab_size=31,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=6,
        num_key_value_heads=2,
        head_dim=head_dim,
        max_position_embeddings=128,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "sdpa"
    sps = PretrainedSPS(Qwen3ForCausalLM(cfg)).eval()
    return sps, QwenSPSMTP(sps, 4).eval()


def test_first_horizon_matches_sps_and_one_block_matches_sequential_verification():
    sps, model = tiny()
    ids = torch.tensor([[2, 4, 8, 6, 3]])
    with torch.no_grad():
        block = model(ids).logits
        torch.testing.assert_close(
            block[:, :, 0], sps(ids).logits, atol=1e-6, rtol=1e-5
        )
        for end in range(1, ids.shape[1] + 1):
            torch.testing.assert_close(
                block[:, end - 1, 0],
                model(ids[:, :end]).logits[:, -1, 0],
                atol=1e-6,
                rtol=1e-5,
            )
        # Later slots cannot influence P1, earlier groups, or real states.
        model.new_embeddings.add_(torch.randn_like(model.new_embeddings) * 2)
        changed = model(ids).logits
        torch.testing.assert_close(
            changed[:, :, 0], block[:, :, 0], atol=1e-6, rtol=1e-5
        )
        assert not torch.allclose(changed[:, :, 1:], block[:, :, 1:])
        altered = ids.clone()
        altered[:, 3:] = torch.tensor([15, 16])
        torch.testing.assert_close(
            model(altered).logits[:, :3], changed[:, :3], atol=1e-6, rtol=1e-5
        )


def test_horizon_loss_does_not_cross_assistant_boundaries_or_sequence_end():
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    assistant = torch.tensor([[0, 1, 1, 0, 1, 1]])
    labels = horizon_targets(tokens, assistant, 4)
    assert labels[0, 0].tolist() == [2, 3, -100, -100]
    assert labels[0, 1].tolist() == [3, -100, -100, -100]
    assert labels[0, 2].tolist() == [-100, -100, -100, -100]
    assert labels[0, 3].tolist() == [5, 6, -100, -100]
    assert labels[0, 4].tolist() == [6, -100, -100, -100]


def test_staged_training_freezes_parent_then_updates_shared_backbone():
    _, model = tiny()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    labels = horizon_targets(tokens, torch.ones_like(tokens), 4)
    parent = {k: v.detach().clone() for k, v in model.backbone.state_dict().items()}
    first = model.prediction.detach().clone()
    model.set_stage(True)
    value, count = model(tokens[:, :-1], targets=labels)
    value.backward()
    assert count == 6
    assert model.new_embeddings.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.parameters())
    assert model.prediction.grad is None
    model.zero_grad()
    model.set_stage(False)
    value, count = model(tokens[:, :-1], targets=labels)
    value.backward()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.step()
    assert count == 10
    assert not torch.equal(first, model.prediction)
    assert any(
        not torch.equal(v, parent[k]) for k, v in model.backbone.state_dict().items()
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("autocast", [False, True])
def test_flex_training_backend_matches_dense_qwen_mask_and_has_gradients(autocast):
    _, model = tiny(head_dim=16)
    model = model.cuda()
    tokens = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=autocast
    ):
        expected = model(tokens).logits
    model.backend = "flex_attention"
    model.backbone.set_attn_implementation("sps_flex_attention")
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=autocast
    ):
        tolerance = 0.005 if autocast else 1e-5
        torch.testing.assert_close(
            model(tokens).logits, expected, atol=tolerance, rtol=tolerance * 10
        )
    labels = horizon_targets(tokens, torch.ones_like(tokens), 4)
    for frozen in (True, False):
        model.zero_grad(set_to_none=True)
        model.set_stage(frozen)
        if not frozen:
            model.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            loss, _ = model(tokens[:, :-1], targets=labels)
        loss.backward()
        assert torch.isfinite(model.new_embeddings.grad).all()
        assert model.new_embeddings.grad.abs().sum() > 0
        assert all(p.dtype == torch.float32 for p in model.parameters())


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n,h,hkv,d", [(4, 6, 2, 8), (4, 16, 8, 128), (1, 16, 8, 128)])
def test_paged_prediction_kernel_matches_dense_mask_with_reordered_pages(
    n, h, hkv, d, monkeypatch
):
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm-sps/vllm/v1/attention/ops/sps_prediction.py"
    )
    if not path.exists():
        pytest.skip("Sibling vLLM worktree is absent")
    spec = importlib.util.spec_from_file_location("sps_prediction_kernel", path)
    module = importlib.util.module_from_spec(spec)
    # Use the real Triton kernel in the SPS environment without importing
    # the incompatible vLLM engine. Only its Triton re-export is substituted.
    import sys
    from types import SimpleNamespace

    import triton
    import triton.language as tl

    monkeypatch.setitem(
        sys.modules, "vllm.triton_utils", SimpleNamespace(triton=triton, tl=tl)
    )
    spec.loader.exec_module(module)
    torch.manual_seed(11)
    t = 3
    q = torch.randn(t, n, h, d, device="cuda", dtype=torch.bfloat16)
    pk = torch.randn(t, n, hkv, d, device="cuda", dtype=torch.bfloat16)
    pv = torch.randn_like(pk)
    real_k = torch.randn(160, hkv, d, device="cuda", dtype=torch.bfloat16)
    real_v = torch.randn_like(real_k)
    cache = torch.zeros(10, hkv, 16, 2 * d, device="cuda", dtype=torch.bfloat16)
    pages = [4, 9, 1, 7, 0, 2, 5, 6, 8, 3]
    table = torch.tensor([pages], device="cuda", dtype=torch.int32)
    for i in range(160):
        cache[pages[i // 16], :, i % 16, :d] = real_k[i]
        cache[pages[i // 16], :, i % 16, d:] = real_v[i]
    positions = torch.tensor([63, 64, 129], device="cuda", dtype=torch.int64)
    output = module.sps_prediction_attention(q, pk, pv, cache, table, positions)
    expected = torch.empty_like(output)
    for i in range(t):
        for slot in range(n):
            keys = torch.cat((real_k[: int(positions[i]) + 1], pk[i, : slot + 1]))
            values = torch.cat((real_v[: int(positions[i]) + 1], pv[i, : slot + 1]))
            expected[i, slot] = torch.nn.functional.scaled_dot_product_attention(
                q[i, slot][None, :, None],
                keys.transpose(0, 1)[None],
                values.transpose(0, 1)[None],
                enable_gqa=True,
            )[0, :, 0]
    torch.testing.assert_close(output, expected, atol=0.025, rtol=0.025)
