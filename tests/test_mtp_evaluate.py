"""Small CPU checks for evaluation cache recovery, targets and timing contracts."""
from types import SimpleNamespace, MethodType

import pytest
import torch

from modeling.models.sps.core import SPSConfig, SPSModelBase
from modeling.models.sps.mtp import SPSMTP
from modeling.mtp_speculative import speculative_generate
from scripts import mtp_evaluate as ev


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def model(n=3):
    torch.manual_seed(11)
    ref = SPSModelBase(SPSConfig(vocab_size=19, block_size=16, n_layer=1,
        n_head=2, hidden_size=16, intermediate_size=24, eos_token_id=16,
        pad_token_id=17, predict_token_id=18, window_size=0, dropout=0.0))
    # Native-method contract on CPU; the small oracle evaluates exact full logits.
    oracle = SPSMTP(ref, 1, backend='sdpa').eval()
    def prefill(self, ids, budget, **kwargs):
        return SimpleNamespace(input_ids=ids.clone(), budget=budget), oracle(ids)[:, -1, 0], 'cpu_oracle'
    def decode(self, state, token, active):
        return oracle(torch.cat((state.input_ids, token[:, None]), 1))[:, -1, 0]
    def advance(self, state, token, active):
        state.input_ids = torch.cat((state.input_ids, token[:, None]), 1)
    ref._prefill_generation_state = MethodType(prefill, ref)
    ref._decode_generation_state = MethodType(decode, ref)
    ref._advance_decode_state = MethodType(advance, ref)
    return SPSMTP(ref, n, backend='sdpa').eval()


def test_adapter_recovers_rejected_cache_and_matches_full_conditionals():
    mtp = model()
    adapter = ev.SpeculativeAdapter(mtp, temperature=.7)
    prefix = torch.tensor([1, 2, 3])
    with torch.inference_mode():
        q = adapter.propose(prefix)
        expected = ev.probabilities(mtp(prefix[None])[0, -1], mtp.config, .7)
        torch.testing.assert_close(q, expected)
        draft = torch.tensor([5, 6, 7])
        p = adapter.verify(prefix, draft)
        for j in range(4):
            full = torch.cat((prefix, draft[:j]))
            expected = ev.probabilities(mtp(full[None])[0, -1, :1], mtp.config, .7)[0]
            torch.testing.assert_close(p[j], expected, atol=2e-5, rtol=2e-4)
        # Reject draft after one acceptance, then insert a correction token.
        corrected = torch.tensor([1, 2, 3, 5, 9])
        q = adapter.propose(corrected)
        expected = ev.probabilities(mtp(corrected[None])[0, -1], mtp.config, .7)
        torch.testing.assert_close(q, expected, atol=2e-5, rtol=2e-4)
        adapter.finish(corrected)
        assert torch.equal(adapter.verifier.prefix, corrected)
        # Same prefix, shorter prefix, and EOS boundary all recover correctly.
        for ids in (corrected, prefix, torch.tensor([1, 16, 3, 4])):
            actual = adapter.propose(ids)
            expected = ev.probabilities(mtp(ids[None])[0, -1], mtp.config, .7)
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)


def test_speculative_greedy_matches_full_reference_across_short_budgets():
    mtp = model()
    prefix = torch.tensor([1, 2])
    for budget in (0, 1, 2, 7):
        adapter = ev.SpeculativeAdapter(mtp)
        result = speculative_generate(prefix, adapter.propose, adapter.verify,
            max_new_tokens=budget, eos_token_id=16)
        expected = []
        with torch.inference_mode():
            for _ in range(budget):
                ids = torch.cat((prefix, torch.tensor(expected, dtype=torch.long)))
                token = ev.probabilities(mtp(ids[None])[0, -1, :1], mtp.config).argmax(-1).item()
                expected.append(token)
                if token == 16:
                    break
        assert result.token_ids.tolist() == expected


def test_horizon_metrics_masks_boundaries_and_uses_bounded_chunks(monkeypatch):
    mtp = model()
    ids = torch.tensor([1, 2, 16, 3, 4, 5, 6, 7, 8, 9, 10])
    class Data:
        tokens = ids
        def __getitem__(self, start):
            return ids[start:start + 8 + 3]
    monkeypatch.setattr(ev, 'TokenSequences', lambda *a, **kw: Data())
    monkeypatch.setattr(ev, 'MTP_CONTEXT', 8)
    monkeypatch.setattr(ev, 'validation_starts', lambda *a: [0])
    monkeypatch.setattr(ev, 'METRIC_CHUNK', 2)
    mtp.train()
    metrics = ev.horizon_metrics(mtp, 'cpu', total_tokens=8)
    assert mtp.training
    with torch.inference_mode():
        logits = mtp(ids[:8][None])[0]
        targets = ev.shifted_targets(ids[None], 8, 3, eos=16, pad=17)[0]
        for j, row in enumerate(metrics):
            valid = targets[:,j] != -100
            loss = torch.nn.functional.cross_entropy(logits[:,j], targets[:,j], ignore_index=-100, reduction='sum')
            assert row['count'] == valid.sum().item()
            assert row['nll'] == pytest.approx((loss / valid.sum()).item(), rel=1e-5)
            assert row['accuracy'] == pytest.approx(((logits[:,j].argmax(-1) == targets[:,j]) & valid).sum().item() / valid.sum().item())


def test_baseline_calls_original_generate_and_strips_padding_after_eos():
    class Reference:
        config = SimpleNamespace(eos_token_id=6, pad_token_id=7)
        calls = []
        def _prefill_generation_state(self, *args, **kwargs):
            return None
        def generate(self, ids, budget, **kwargs):
            self.calls.append((budget, kwargs))
            self._prefill_generation_state(ids, budget, require_batched=True)
            return torch.cat((ids, torch.tensor([[4,6,7,7]])), 1)
    ref = Reference()
    row = ev.benchmark_reference(ref, torch.tensor([1,2]), mode='sample', max_new_tokens=4, seed=5)
    assert row['token_ids'] == [4,6]
    assert row['tokens'] == 2
    assert row['total_seconds'] >= row['prefill_seconds'] >= 0
    assert ref.calls[0][1] == dict(do_sample=True, temperature=ev.MTP_TEMPERATURE,
                                  top_k=None, top_p=None, stop_on_eos=True)
    assert '_prefill_generation_state' not in ref.__dict__


def test_quality_restores_training_backend_and_checks_raw_cache_logits(monkeypatch):
    mtp = model()
    mtp.train()
    monkeypatch.setattr(ev, '_horizon_metrics', lambda model, device, count: [model.backend])
    mtp.backend = 'auto'
    assert ev.horizon_metrics(mtp, 'cpu', 8) == ['sdpa']
    assert mtp.backend == 'auto'
    assert mtp.training
    mtp.backend = 'sdpa'
    result = ev.check_cached_logits(mtp, torch.tensor([1,2,3]))
    assert result['passed']
    assert result['maximum_proposal_logit_error'] < 2e-4
    assert result['maximum_reference_logit_error'] == 0


def test_timed_speculation_records_actual_emitted_tokens_and_counters():
    mtp = model()
    prefix = torch.tensor([1,2])
    for mode in ('greedy', 'sample'):
        row = ev.benchmark_speculative(mtp, prefix, mode=mode, max_new_tokens=5)
        assert row['tokens'] == len(row['token_ids'])
        assert 1 <= row['tokens'] <= 5
        assert len(row['accepted_by_horizon']) == 3
        assert len(row['proposed_by_horizon']) == 3
        assert row['total_seconds'] >= row['prefill_seconds'] >= 0
        assert row['decode_seconds'] > 0
        assert row['tokens_per_second'] > 0
        assert row['rounds'] > 0


def test_cli_uses_validated_trained_marker_checkpoint(monkeypatch, tmp_path):
    import json
    import experiments
    exp = SimpleNamespace(name='chosen', horizons=3)
    checkpoint = tmp_path/'adaptation_final.pt'
    checkpoint.touch()
    (tmp_path/'trained.json').write_text(json.dumps(dict(checkpoint=str(checkpoint), design_signature='current')))
    monkeypatch.setattr(experiments, 'SUITE', 'mtp')
    monkeypatch.setattr(experiments, 'collect_experiments', lambda pattern: [exp])
    monkeypatch.setattr(experiments, 'design_signature', lambda: 'current')
    monkeypatch.setattr(ev, 'output_dir', lambda exp: tmp_path)
    called = []
    monkeypatch.setattr(ev, 'evaluate', lambda exp, checkpoint: called.append((exp, checkpoint)))
    ev.main(['--filter', 'chosen'])
    assert called == [(exp, checkpoint)]
    monkeypatch.setattr(experiments, 'design_signature', lambda: 'changed')
    with pytest.raises(RuntimeError, match='Stale'):
        ev.main(['--filter', 'chosen'])


def test_native_verifier_branch_snapshot_matches_source_after_eos_and_rejection():
    mtp = model()
    verifier = ev.NativePrefixVerifier(mtp.reference, max_new_tokens=8)
    prefix = torch.tensor([1, 2])
    first = verifier.ensure(prefix).clone()
    state = verifier.state
    draft = torch.tensor([16, 3, 4])
    expected = []
    with torch.inference_mode():
        for j in range(4):
            ids = torch.cat((prefix, draft[:j]))
            expected.append(mtp(ids[None])[0, -1, 0])
        actual = verifier.verify_logits(prefix, draft)
        torch.testing.assert_close(actual, torch.stack(expected), atol=2e-5, rtol=2e-4)
        assert verifier.state is state
        assert verifier.prefix.tolist() == [1, 2]
        torch.testing.assert_close(verifier.ensure(prefix), first, atol=0, rtol=0)
        # The rejected branch crossed EOS. The replacement retains no branch state.
        corrected = torch.tensor([1, 2, 9])
        actual = verifier.ensure(corrected)
        expected = mtp(corrected[None])[0, -1, 0]
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        assert verifier.state.input_ids.tolist() == [[1, 2, 9]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native SPS generator requires CUDA')
def test_real_native_cuda_verifier_matches_original_generator_and_probabilities():
    from modeling.models.sps import SPSModel
    torch.manual_seed(37)
    reference = SPSModel(SPSConfig(vocab_size=19, block_size=32, n_layer=1,
        n_head=1, hidden_size=64, intermediate_size=128, eos_token_id=16,
        pad_token_id=17, predict_token_id=18, window_size=0, dropout=0.0)).cuda().eval()
    mtp = SPSMTP(reference, 3, backend='sdpa').cuda().eval()
    prefix = torch.tensor([1, 2, 16, 3], device='cuda')
    with torch.inference_mode():
        base = ev.benchmark_reference(reference, prefix, mode='greedy', max_new_tokens=8)
        actual = ev.benchmark_speculative(mtp, prefix, mode='greedy', max_new_tokens=8)
        assert actual['token_ids'] == base['token_ids']
        assert actual['verifier_backend'] == 'native_sequential'
        adapter = ev.SpeculativeAdapter(mtp, max_new_tokens=8)
        draft = torch.tensor([16, 4, 5], device='cuda')
        probabilities = adapter.verify(prefix, draft)
        state, logits, _ = reference._prefill_generation_state(prefix[None], 8, require_batched=True)
        expected = [ev.probabilities(logits[0], mtp.config)]
        active = torch.ones(1, dtype=torch.bool, device='cuda')
        for token in draft:
            logits = reference._decode_generation_state(state, token[None], active)
            reference._advance_decode_state(state, token[None], active)
            expected.append(ev.probabilities(logits[0], mtp.config))
        torch.testing.assert_close(probabilities, torch.stack(expected), atol=0, rtol=0)
        corrected = torch.cat((prefix, torch.tensor([9], device='cuda')))
        cached = adapter.verifier.ensure(corrected)
        state, _, _ = reference._prefill_generation_state(prefix[None], 8, require_batched=True)
        oracle = reference._decode_generation_state(state, corrected[-1:], active)
        torch.testing.assert_close(cached, oracle[0], atol=0, rtol=0)
