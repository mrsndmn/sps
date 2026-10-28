"""Reference-exact speculative generation from fixed-prefix MTP probabilities.

Callbacks own model execution, probability transforms (temperature/top-k/etc.)
and caches. ``propose(prefix)`` returns [N,V] probabilities, all conditioned on
that *same* prefix. Its rows are sampled independently. ``verify(prefix, draft)``
returns [len(draft)+1,V] reference probabilities: row j conditions on prefix plus
``draft[:j]``. The last row is the bonus-token distribution. This distinction
is essential: later MTP rows do not condition on earlier sampled draft tokens.

Each callback must reconcile its cache to the supplied complete accepted prefix;
a verifier may have cached rejected draft tokens. Callbacks must not mutate their
arguments. On return, reconcile caches to ``cat(prefix, result.token_ids)`` before
reusing them: the last callback can have advanced beyond a rejection or EOS.
Timing belongs around this function and callback/cache work in its caller.
"""
from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import Tensor


@dataclass(frozen=True)
class SpeculativeResult:
    """Generated tokens (excluding prompt), with zero-based horizon counters.

    Proposed counts include every sampled/argmax draft row, including rows beyond
    a rejection or EOS. Accepted counts include only draft tokens actually emitted.
    Bonus and correction tokens contribute to neither accepted counter.
    reached_by_horizon counts slots whose acceptance test actually ran; it excludes
    slots discarded after an earlier rejection or EOS.
    """

    token_ids: Tensor
    accepted_by_horizon: tuple[int, ...]
    proposed_by_horizon: tuple[int, ...]
    rounds: int
    reached_by_horizon: tuple[int, ...] = ()


def _probabilities(value: Tensor, name: str, device: torch.device,
                   rows: int | None = None, vocabulary: int | None = None) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise ValueError(f'{name} must return a [rows, vocabulary] probability tensor')
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f'{name} probability dimensions must be positive')
    if rows is not None and value.shape[0] != rows:
        raise ValueError(f'{name} must return {rows} probability rows')
    if vocabulary is not None and value.shape[1] != vocabulary:
        raise ValueError(f'{name} vocabulary must match proposal vocabulary')
    if value.device != device or not value.is_floating_point():
        raise ValueError(f'{name} probabilities must be floating point on prefix device')
    # Use at least float32 for acceptance/residual arithmetic. Normalize the
    # validated rows so sampling and the p/q calculation use identical masses.
    value = value.to(dtype=torch.float64 if value.dtype == torch.float64 else torch.float32)
    totals = value.sum(dim=-1, keepdim=True)
    if (not torch.isfinite(value).all() or (value < 0).any()
            or not torch.allclose(totals, torch.ones_like(totals), rtol=1e-4, atol=1e-5)):
        raise ValueError(f'{name} probabilities must be finite, nonnegative and sum to one')
    return value / totals


@torch.inference_mode()
def speculative_generate(
    prefix: Tensor,
    propose: Callable[[Tensor], Tensor],
    verify: Callable[[Tensor, Tensor], Tensor],
    *,
    max_new_tokens: int,
    mode: Literal['greedy', 'sample'] = 'greedy',
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
    draft_proposal: Callable | None = None,
) -> SpeculativeResult:
    """Generate a single sequence, preserving the verifier's decoding law.

    Input ``prefix`` is a nonempty 1-D integer tensor. Outputs share its device
    and use int64 IDs. The generator must match the probability tensor device.
    A prompt ending in EOS is allowed; only newly emitted EOS terminates output.
    The horizon count and vocabulary must remain constant across rounds.
    A zero output budget does not call either callback.

    Optional draft_proposal(history, remaining, mode, generator) returns sampled
    token IDs [N] and the actual proposal probabilities [N,V] conditioned on those
    preceding draft IDs. It must use the supplied generator for stochastic draws.
    Keep N fixed and fill unused rows with any valid dummy distribution when the
    remaining budget is shorter; those rows are truncated before verification.
    With no draft_proposal, independent fixed-prefix MTP sampling is unchanged.
    """
    if (not isinstance(prefix, Tensor) or prefix.ndim != 1 or prefix.numel() == 0
            or prefix.dtype not in (torch.int32, torch.int64)):
        raise ValueError('prefix must be a nonempty one-dimensional integer tensor')
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 0:
        raise ValueError('max_new_tokens must be a nonnegative integer')
    if mode not in ('greedy', 'sample'):
        raise ValueError("mode must be 'greedy' or 'sample'")
    history = prefix.to(dtype=torch.long).clone()
    initial_length = history.numel()
    accepted: list[int] = []
    proposed: list[int] = []
    reached: list[int] = []
    vocabulary = None
    rounds = 0

    def emit(token: Tensor) -> bool:
        nonlocal history
        history = torch.cat((history, token.reshape(1).to(dtype=torch.long)))
        return ((eos_token_id is not None and token.item() == eos_token_id)
                or history.numel() - initial_length == max_new_tokens)

    while history.numel() - initial_length < max_new_tokens:
        remaining = max_new_tokens - (history.numel() - initial_length)
        prepared=None
        if draft_proposal is None:
            proposal=propose(history)
        else:
            prepared,proposal=draft_proposal(history,remaining,mode,generator)
        q = _probabilities(proposal, 'proposal', history.device,
                           rows=len(accepted) or None, vocabulary=vocabulary)
        if not accepted:
            accepted = [0] * q.shape[0]
            proposed = [0] * q.shape[0]
            reached = [0] * q.shape[0]
            vocabulary = q.shape[1]
        if prepared is not None:
            if (prepared.shape!=(q.shape[0],) or prepared.device!=q.device
                    or prepared.dtype!=torch.long or bool(((prepared<0)|(prepared>=vocabulary)).any())):
                raise ValueError('Invalid conditional draft tokens')
            if bool((q.gather(1,prepared[:,None])<=0).any()):
                raise ValueError('Conditional draft has zero proposal probability')
            draft=prepared[:remaining]
            q=q[:remaining]
        else:
            q=q[:remaining]
            draft = (q.argmax(dim=-1) if mode == 'greedy'
                     else torch.multinomial(q, 1, generator=generator).squeeze(-1))
        p = _probabilities(verify(history, draft), 'verification', history.device,
                           rows=len(draft) + 1, vocabulary=vocabulary)
        rounds += 1
        for j in range(len(draft)):
            proposed[j] += 1
        rejected = False
        for j, token in enumerate(draft):
            reached[j] += 1
            if mode == 'greedy':
                accept = bool(token == p[j].argmax())
            else:
                # q(token)>0 because the token was drawn from q. p(token)=0
                # therefore rejects without division-by-zero corner cases.
                ratio = (p[j, token] / q[j, token]).clamp(max=1)
                accept = bool(torch.rand((), device=q.device, generator=generator) < ratio)
            if accept:
                accepted[j] += 1
                if emit(token):
                    return SpeculativeResult(history[initial_length:], tuple(accepted), tuple(proposed), rounds, tuple(reached))
            else:
                if mode == 'greedy':
                    correction = p[j].argmax()
                else:
                    residual = (p[j] - q[j]).clamp(min=0)
                    # Rejection has positive probability only when residual
                    # mass is positive. Never fall back to an incorrect law.
                    mass = residual.sum()
                    if not mass > 0:
                        raise RuntimeError('rejection produced zero residual probability mass')
                    correction = torch.multinomial(residual / mass, 1, generator=generator).squeeze(0)
                if emit(correction):
                    return SpeculativeResult(history[initial_length:], tuple(accepted), tuple(proposed), rounds, tuple(reached))
                rejected = True
                break
        if not rejected:
            bonus = (p[-1].argmax() if mode == 'greedy'
                     else torch.multinomial(p[-1], 1, generator=generator).squeeze(0))
            if emit(bonus):
                break
    return SpeculativeResult(history[initial_length:], tuple(accepted), tuple(proposed), rounds, tuple(reached))
