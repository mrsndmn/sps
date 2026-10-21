"""Exact and empirical checks for fixed-prefix MTP speculative decoding."""
from collections import Counter
from itertools import product

import pytest
import torch

from src.modeling.mtp_speculative import speculative_generate


def reference(prefix, proposals):
    # A nontrivial Markov reference: row j sees all earlier real proposals.
    history = prefix.tolist()
    rows = []
    for token in [*proposals.tolist(), None]:
        rows.append([0.75, 0.25] if history[-1] == 0 else [0.25, 0.75])
        if token is not None:
            history.append(token)
    return torch.tensor(rows, dtype=torch.float64, device=prefix.device)


@pytest.mark.parametrize('horizons', [1, 2, 3, 4, 7])
@pytest.mark.parametrize('budget', [1, 2, 5, 9])
def test_greedy_matches_reference_with_rejection_and_budget(horizons, budget):
    seen = []

    def propose(prefix):
        seen.append(prefix.tolist())
        return torch.tensor([[0.1, 0.9]] * horizons)

    result = speculative_generate(torch.tensor([0]), propose, reference, max_new_tokens=budget)
    assert result.token_ids.tolist() == [0] * budget
    assert seen == [[0] * (i + 1) for i in range(budget)]
    assert result.rounds == budget
    assert sum(result.accepted_by_horizon) == 0
    assert len(result.proposed_by_horizon) == horizons
    assert result.proposed_by_horizon[0] == budget


def test_all_accepted_bonus_and_counters():
    result = speculative_generate(
        torch.tensor([1]), lambda p: torch.tensor([[0., 1.]] * 3),
        reference, max_new_tokens=6,
    )
    assert result.token_ids.tolist() == [1] * 6
    assert result.rounds == 2
    assert result.accepted_by_horizon == (2, 2, 1)
    assert result.proposed_by_horizon == (2, 2, 1)


@pytest.mark.parametrize('mode', ['greedy', 'sample'])
@pytest.mark.parametrize('eos', [0, 1])
def test_eos_on_rejection_acceptance_or_bonus(mode, eos):
    def verify(prefix, proposals):
        return torch.tensor([[1. - eos, float(eos)]] * (len(proposals) + 1))

    for draft in [eos, 1 - eos]:
        result = speculative_generate(
            torch.tensor([5]), lambda p: torch.tensor([[1. - draft, float(draft)]] * 4),
            verify, max_new_tokens=20, mode=mode, eos_token_id=eos,
        )
        assert result.token_ids.tolist() == [eos]
        assert result.rounds == 1
    result = speculative_generate(
        torch.tensor([5]), lambda p: torch.tensor([[float(eos), 1. - eos]]),
        lambda p, q: torch.tensor([[float(eos), 1. - eos], [1. - eos, float(eos)]]),
        max_new_tokens=20, mode=mode, eos_token_id=eos,
    )
    assert result.token_ids.tolist() == [1 - eos, eos]


def test_zero_budget_does_not_invoke_callbacks():
    def fail(*args):
        raise AssertionError('callbacks must not run')
    result = speculative_generate(torch.tensor([0]), fail, fail, max_new_tokens=0)
    assert result.token_ids.tolist() == []
    assert result.rounds == 0
    assert result.proposed_by_horizon == ()


def test_sampling_reproducible_and_matches_reference_joint_empirically():
    def draft(prefix):
        return torch.tensor([[0.5, 0.5], [0.9, 0.1]], dtype=torch.float64)
    def run(seed):
        return speculative_generate(torch.tensor([0]), draft, reference,
                                    max_new_tokens=2, mode='sample',
                                    generator=torch.Generator().manual_seed(seed)).token_ids.tolist()
    assert run(42) == run(42)
    counts = Counter(tuple(run(i)) for i in range(4000))
    expected = {(0, 0): .75 * .75, (0, 1): .75 * .25,
                (1, 0): .25 * .25, (1, 1): .25 * .75}
    for sequence, probability in expected.items():
        assert abs(counts[sequence] / 4000 - probability) < .035


def test_exact_single_token_distribution_with_disjoint_and_overlapping_support(monkeypatch):
    # Enumerate categorical proposals, rejection corrections and U~Uniform grid.
    # These rational examples have all acceptance thresholds on this grid.
    for q, p in [([1., 0.], [0., 1.]), ([.5, .5], [.75, .25]),
                 ([.25, .75], [.5, .5])]:
        mass = [0., 0.]
        for proposal, u_index, correction in product(range(2), range(12), range(2)):
            if q[proposal] == 0:
                continue
            weight = q[proposal] / 12
            residual = [max(pi - qi, 0.) for pi, qi in zip(p, q)]
            total = sum(residual)
            correction_probability = residual[correction] / total if total else .5
            weight *= correction_probability
            calls = 0
            def sample(probs, num_samples, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    assert torch.allclose(probs.reshape(-1), torch.tensor(q, dtype=probs.dtype))
                    return torch.tensor([[proposal]])
                assert torch.allclose(probs, torch.tensor(residual, dtype=probs.dtype) / total)
                return torch.tensor([correction])
            monkeypatch.setattr(torch, 'multinomial', sample)
            monkeypatch.setattr(torch, 'rand', lambda *a, **k: torch.tensor((u_index + .5) / 12))
            result = speculative_generate(
                torch.tensor([0]), lambda _: torch.tensor([q], dtype=torch.float64),
                lambda _, tokens: torch.tensor([p, p], dtype=torch.float64),
                max_new_tokens=1, mode='sample',
            )
            mass[result.token_ids.item()] += weight
        assert mass == pytest.approx(p)


@pytest.mark.parametrize('bad', [torch.ones(2), torch.ones(0, 2), torch.tensor([[0., 0.]]),
                                torch.tensor([[-1., 2.]]), torch.tensor([[float('nan'), 1.]]),
                                torch.tensor([[.3, .3]])])
def test_invalid_proposal_probabilities(bad):
    with pytest.raises(ValueError, match='propos'):
        speculative_generate(torch.tensor([0]), lambda _: bad, reference, max_new_tokens=2)


@pytest.mark.parametrize('bad', [torch.ones(2, 3) / 3, torch.ones(1, 2) / 2,
                                torch.tensor([[0., 0.]] * 3)])
def test_invalid_verification_shape_or_probabilities(bad):
    with pytest.raises(ValueError, match='verif'):
        speculative_generate(torch.tensor([0]), lambda _: torch.ones(2, 2) / 2,
                             lambda p, q: bad, max_new_tokens=2)


@pytest.mark.parametrize('kwargs', [{'max_new_tokens': -1}, {'max_new_tokens': 1, 'mode': 'bogus'}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        speculative_generate(torch.tensor([0]), lambda _: None, reference, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_generator_and_output_device():
    result = speculative_generate(torch.tensor([0], device='cuda'),
        lambda p: torch.tensor([[.5, .5]] * 2, device='cuda'), reference,
        max_new_tokens=5, mode='sample', generator=torch.Generator(device='cuda').manual_seed(4))
    assert result.token_ids.device.type == 'cuda'
    assert len(result.token_ids) == 5


def test_partial_rejection_restarts_from_accepted_and_corrected_prefix():
    prefixes = []
    verified = []
    prompt = torch.tensor([7])

    def propose(prefix):
        prefixes.append(prefix.tolist())
        return (torch.tensor([[1., 0.], [0., 1.], [0., 1.]]) if len(prefix) == 1
                else torch.tensor([[1., 0.]] * 3))

    def verify(prefix, draft):
        verified.append((prefix.tolist(), draft.tolist()))
        return torch.tensor([[1., 0.]] * (len(draft) + 1))

    result = speculative_generate(prompt, propose, verify, max_new_tokens=4)
    assert result.token_ids.tolist() == [0, 0, 0, 0]
    assert prompt.tolist() == [7]
    assert prefixes == [[7], [7, 0, 0]]
    assert verified == [([7], [0, 1, 1]), ([7, 0, 0], [0, 0])]
    assert result.accepted_by_horizon == (2, 1, 0)
    assert result.proposed_by_horizon == (2, 2, 1)


def test_horizon_count_cannot_change_between_rounds():
    def propose(prefix):
        return torch.tensor([[1., 0.]] * (1 if len(prefix) == 1 else 2))
    with pytest.raises(ValueError, match='proposal.*rows'):
        speculative_generate(torch.tensor([0]), propose, reference, max_new_tokens=4)
