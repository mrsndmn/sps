"""CPU wiring + masking unit tests for the multi-token-ahead probe analysis.

Uses a tiny randomly-initialised standard model (flex attention, CPU-capable) to exercise
the batcher, k-ahead document masking, stream extraction, head training, and eval plumbing --
no GPU and no checkpoint download.
"""

import numpy as np
import torch

from modeling.models.full_attention_model import Model, ModelConfig
from scripts.analysis import multitoken_head_nll as M


def test_masking():
    """k-ahead mask must (a) drop eos/pad sources and (b) block targets across an eos boundary."""
    EOS, PAD = M.EOS_TOKEN_ID, M.PAD_TOKEN_ID
    a, b, c, d, e, f = 10, 11, 12, 13, 14, 15
    toks = torch.tensor([[a, b, EOS, c, d, e, f]], dtype=torch.long)  # doc0=[a,b,EOS], doc1=[c,d,e,f]
    seqlen, k_max = 4, 2
    x, targets = M.build_inputs_and_targets(toks, seqlen, k_max)
    assert x.tolist() == [[a, b, EOS, c]]
    I = M.IGNORE_INDEX
    # k=1: pos0->b (ok), pos1->EOS (ok, end-of-doc target), pos2 source=EOS (masked), pos3->d (ok, same doc)
    assert targets[1].tolist() == [[b, EOS, I, d]], targets[1].tolist()
    # k=2: pos0->EOS (ok), pos1->c is PAST the eos boundary (masked), pos2 source=EOS (masked), pos3->e (ok)
    assert targets[2].tolist() == [[EOS, I, I, e]], targets[2].tolist()


def test_cpu_wiring():
    torch.manual_seed(0)
    cfg = ModelConfig(block_size=128, vocab_size=50304, n_layer=2, n_head=2, hidden_size=32,
                      intermediate_size=64, use_triton_full_attention=False)
    model = Model(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    device = torch.device("cpu")
    hidden, vocab = cfg.hidden_size, cfg.vocab_size
    init_w = model.transformer.wte.weight.detach().clone().float()
    heads = M.ProbeHeads(hidden, vocab, M.streams_for_kind("full_attention"), k_max=4, init_weight=init_w)

    # tie intact and head is a *copy* (not aliased) of wte
    assert model.lm_head.weight is model.transformer.wte.weight
    assert torch.equal(heads.head("main", 1).weight, model.transformer.wte.weight)
    assert heads.head("main", 1).weight is not model.transformer.wte.weight

    # fake corpus: random tokens with sprinkled eos
    rng = np.random.default_rng(0)
    data = rng.integers(0, 50256, size=20000).astype(np.uint16)
    data[rng.integers(0, 20000, size=200)] = M.EOS_TOKEN_ID

    seqlen, k_max, bs = 16, 4, 4
    opt = torch.optim.AdamW(heads.parameters(), lr=1e-2)

    wte_before = model.transformer.wte.weight.detach().clone()
    head_before = heads.head("main", 1).weight.detach().clone()

    losses = []
    for _ in range(25):
        toks = M.sample_windows(data, bs, seqlen, k_max, rng).to(device)
        x, targets = M.build_inputs_and_targets(toks, seqlen, k_max)
        feats = M.extract_streams(model, "full_attention", x)
        assert feats["main"].shape == (bs, seqlen, hidden), feats["main"].shape
        assert not feats["main"].requires_grad  # detached leaf
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for k in range(1, k_max + 1):
            logits = heads.head("main", k)(feats["main"]).float()
            tgt = targets[k]
            cnt = (tgt != M.IGNORE_INDEX).sum().clamp(min=1)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab), tgt.view(-1), ignore_index=M.IGNORE_INDEX, reduction="sum") / cnt
            loss.backward()
            step_loss += float(loss)
        opt.step()
        losses.append(step_loss)

    # heads learned; backbone (incl. tied wte/lm_head) untouched
    assert not torch.equal(heads.head("main", 1).weight, head_before), "head did not update"
    assert torch.equal(model.transformer.wte.weight, wte_before), "frozen wte changed!"
    assert model.transformer.wte.weight.grad is None, "grad leaked into frozen backbone"
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"

    # eval plumbing produces the expected (stream,k) keys
    per_batch, native_k1 = M.evaluate(model, "full_attention", heads, data, seqlen=seqlen, k_max=k_max,
                                      batch_size=bs, eval_tokens=2 * bs * seqlen, device=device, seed=1)
    assert set(per_batch.keys()) == {("main", k) for k in range(1, k_max + 1)}
    assert native_k1 > 0
