"""Held-out horizon quality and reference-verified, batch-one generation timing.

The baseline is the source model's existing cached ``generate`` implementation.
All proposal, reference and cache recovery work is inside speculative timing.
"""
from __future__ import annotations

import json
from copy import deepcopy
import math
import time

import torch
from torch.nn import functional as F

from experiments import (MTP_INFERENCE_BACKEND, MTP_CONTEXT, MTP_EVAL_PROMPTS,
    MTP_NEW_TOKENS, MTP_PROMPT_TOKENS, MTP_SEED, MTP_SOURCE,
    MTP_TEMPERATURE, MTP_VAL_TOKENS, output_dir)
from scripts.mtp_common import (TokenSequences, load_mtp, shifted_targets,
                                validation_starts, restore_draft, load_reference)
from modeling.models.sps.mtp import truncate_cache
from modeling.mtp_speculative import speculative_generate

# Bound vocabulary projections during validation; every real prefix is evaluated.
METRIC_CHUNK = 32


def probabilities(logits, config, temperature=MTP_TEMPERATURE):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    transformed = logits.float().clone() / temperature
    transformed[..., config.predict_token_id] = -torch.inf
    transformed[..., config.pad_token_id] = -torch.inf
    return transformed.softmax(-1)


class PrefixCache:
    """Own real-only KV state and recover a prefix after speculative rejection."""
    def __init__(self, model):
        self.model = model
        self.cache = None
        self.last_logits = None

    def ensure(self, prefix):
        previous = 0
        if self.cache is not None:
            limit = min(self.cache.length, prefix.numel())
            equal = self.cache.input_ids[0, :limit] == prefix[:limit]
            wrong = (~equal).nonzero().flatten()
            previous = int(wrong[0]) if wrong.numel() else limit
            if previous == self.cache.length == prefix.numel():
                return self.last_logits
            # To obtain logits after shortening, replay the final retained token.
            previous = min(previous, prefix.numel() - 1)
            self.cache = truncate_cache(self.cache, previous) if previous else None
        self.cache, logits = self.model.encode(prefix[None, previous:], self.cache,
                                               positions=[prefix.numel()-1])
        self.last_logits = logits[0, -1]
        return self.last_logits

    def append(self, tokens):
        self.cache, logits = self.model.encode(tokens[None], self.cache)
        self.last_logits = logits[0, -1]
        return logits[0]


class NativePrefixVerifier:
    """Use exactly the native SPS cached decoding distribution.

    Verification is sequential. A temporary branch copies the native cache,
    including document/reset state. The accepted base stays intact after a
    rejection; appending the actual continuation replays native decode steps.
    Both snapshot and replay are real overhead included by the caller's timer.
    """
    def __init__(self, reference, max_new_tokens=MTP_NEW_TOKENS):
        self.reference = reference
        self.max_new_tokens = max_new_tokens
        self.state = None
        self.prefix = None
        self.last_logits = None

    @torch.inference_mode()
    def ensure(self, prefix):
        extends = (self.prefix is not None and prefix.numel() >= self.prefix.numel()
                   and torch.equal(prefix[:self.prefix.numel()], self.prefix))
        if not extends:
            budget = min(self.max_new_tokens, self.reference.config.block_size-prefix.numel())
            self.state, logits, _ = self.reference._prefill_generation_state(
                prefix[None], budget, require_batched=True)
            self.last_logits = logits[0]
        else:
            active = torch.ones(1, dtype=torch.bool, device=prefix.device)
            for token in prefix[self.prefix.numel():]:
                logits = self.reference._decode_generation_state(self.state, token[None], active)
                self.reference._advance_decode_state(self.state, token[None], active)
                self.last_logits = logits[0]
        self.prefix = prefix.detach().clone()
        return self.last_logits

    @torch.inference_mode()
    def verify_logits(self, prefix, draft):
        first = self.ensure(prefix)
        branch = deepcopy(self.state)
        rows = [first]
        active = torch.ones(1, dtype=torch.bool, device=prefix.device)
        for token in draft:
            logits = self.reference._decode_generation_state(branch, token[None], active)
            self.reference._advance_decode_state(branch, token[None], active)
            rows.append(logits[0])
        return torch.stack(rows)


def _native_reference(model, reference):
    if reference is None:
        if getattr(model, 'train_backbone', False) or getattr(model, 'has_adapted_backbone', False):
            raise RuntimeError('Trainable drafts require an independent original native reference')
        return model.reference
    draft_storage = {p.untyped_storage().data_ptr() for p in model.parameters()}
    if any(p.untyped_storage().data_ptr() in draft_storage for p in reference.parameters()):
        raise RuntimeError('Native reference must not share parameter storage with the draft')
    return reference


class SpeculativeAdapter:
    def __init__(self, model, temperature=MTP_TEMPERATURE, max_new_tokens=MTP_NEW_TOKENS, *, reference=None, verifier_factory=NativePrefixVerifier):
        self.model = model
        self.temperature = temperature
        self.proposer = PrefixCache(model)
        self.verifier = verifier_factory(_native_reference(model, reference), max_new_tokens)

    def prefill(self, prefix):
        self.proposer.ensure(prefix)
        self.verifier.ensure(prefix)

    def propose(self, prefix):
        return probabilities(self.proposer.ensure(prefix), self.model.config, self.temperature)

    def verify(self, prefix, draft):
        raw = self.verifier.verify_logits(prefix, draft)
        return probabilities(raw, self.model.config, self.temperature)

    def finish(self, prefix):
        self.proposer.ensure(prefix)
        self.verifier.ensure(prefix)


@torch.inference_mode()
def _horizon_metrics(model, device, total_tokens=MTP_VAL_TOKENS):
    model.eval()
    context = min(MTP_CONTEXT, model.config.block_size)
    data = TokenSequences('val', context, model.horizons)
    starts = validation_starts(len(data.tokens), context, math.ceil(total_tokens/context))
    sums = torch.zeros((model.horizons, 3), dtype=torch.float64, device=device)
    remaining = total_tokens
    for start in starts:
        length = min(context, remaining)
        if length <= 0:
            break
        tokens = data[start][None].to(device)
        targets = shifted_targets(tokens, length, model.horizons,
            eos=model.config.eos_token_id, pad=model.config.pad_token_id)
        cache = None
        for offset in range(0, length, METRIC_CHUNK):
            end = min(length, offset + METRIC_CHUNK)
            cache, logits = model.encode(tokens[:, offset:end], cache)
            for j in range(model.horizons):
                target = targets[:, offset:end, j]
                valid = target != -100
                prediction = logits[:, :, j].float()
                sums[j, 0] += F.cross_entropy(prediction.reshape(-1, prediction.shape[-1]),
                    target.reshape(-1), ignore_index=-100, reduction='sum')
                sums[j, 1] += ((prediction.argmax(-1) == target) & valid).sum()
                sums[j, 2] += valid.sum()
            del logits
        remaining -= length
    return [dict(horizon=j+1, count=int(count), nll=nll/count if count else None,
                 accuracy=correct/count if count else None)
            for j, (nll, correct, count) in enumerate(sums.cpu().tolist())]


@torch.inference_mode()
def horizon_metrics(model, device, total_tokens=MTP_VAL_TOKENS):
    previous_backend = model.backend
    was_training = model.training
    try:
        model.backend = MTP_INFERENCE_BACKEND
        return _horizon_metrics(model, device, total_tokens)
    finally:
        model.backend = previous_backend
        model.train(was_training)


def _sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def _timing(tokens, total, prefill):
    decode = max(total-prefill, 0.0)
    return dict(token_ids=tokens.tolist(), tokens=tokens.numel(), total_seconds=total,
                prefill_seconds=prefill, decode_seconds=decode,
                tokens_per_second=tokens.numel()/total if total else None,
                decode_tokens_per_second=tokens.numel()/decode if decode else None)


@torch.inference_mode()
def benchmark_reference(reference, prefix, *, mode, max_new_tokens=MTP_NEW_TOKENS,
                        seed=MTP_SEED):
    """Time the actual optimized public SPS generator, including its prefill."""
    original = reference._prefill_generation_state
    had_own_method = '_prefill_generation_state' in reference.__dict__
    own_method = reference.__dict__.get('_prefill_generation_state')
    prefill_seconds = 0.0
    def timed_prefill(*args, **kwargs):
        nonlocal prefill_seconds
        _sync(prefix.device)
        start = time.perf_counter()
        value = original(*args, **kwargs)
        _sync(prefix.device)
        prefill_seconds += time.perf_counter()-start
        return value
    reference._prefill_generation_state = timed_prefill
    devices = [prefix.device.index or 0] if prefix.is_cuda else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            _sync(prefix.device)
            start = time.perf_counter()
            result = reference.generate(prefix[None], max_new_tokens,
                do_sample=mode == 'sample', temperature=MTP_TEMPERATURE,
                top_k=None, top_p=None, stop_on_eos=True)
            _sync(prefix.device)
            total = time.perf_counter()-start
    finally:
        if had_own_method:
            reference._prefill_generation_state = own_method
        else:
            del reference._prefill_generation_state
    tokens = result[0, prefix.numel():]
    eos = (tokens == reference.config.eos_token_id).nonzero().flatten()
    if eos.numel():
        tokens = tokens[:int(eos[0])+1]
    return _timing(tokens, total, prefill_seconds)


@torch.inference_mode()
def benchmark_speculative(model, prefix, *, mode, max_new_tokens=MTP_NEW_TOKENS,
                          seed=MTP_SEED, reference=None, verifier_factory=NativePrefixVerifier):
    _sync(prefix.device)
    start = time.perf_counter()
    adapter = SpeculativeAdapter(model, max_new_tokens=max_new_tokens, reference=reference, verifier_factory=verifier_factory)
    adapter.prefill(prefix)
    _sync(prefix.device)
    prefill_seconds = time.perf_counter()-start
    generator = torch.Generator(device=prefix.device).manual_seed(seed)
    result = speculative_generate(prefix, adapter.propose, adapter.verify,
        max_new_tokens=max_new_tokens, mode=mode,
        eos_token_id=model.config.eos_token_id, generator=generator)
    if result.token_ids.numel():
        adapter.finish(torch.cat((prefix, result.token_ids)))
    _sync(prefix.device)
    total = time.perf_counter()-start
    row = _timing(result.token_ids, total, prefill_seconds)
    row.update(verifier_backend=getattr(adapter.verifier, 'backend', 'native_sequential'), rounds=result.rounds, accepted_by_horizon=list(result.accepted_by_horizon),
               proposed_by_horizon=list(result.proposed_by_horizon),
               accepted_tokens_per_round=sum(result.accepted_by_horizon)/result.rounds if result.rounds else 0)
    return row


@torch.inference_mode()
def check_cached_logits(model, prefix, *, reference=None):
    """Report proposal drift; demand exact native target probabilities.

    Proposal cache/full differences are diagnostics, not correctness gates.
    Proposals may be any valid distribution: speculative acceptance uses the
    actual q probabilities, validated by the decoder, and the exact native p.
    Logical cache recovery and causality are tested independently.
    """
    adapter = SpeculativeAdapter(model, max_new_tokens=model.horizons+1, reference=reference)
    reference = adapter.verifier.reference
    max_raw = 0.0
    max_probability = 0.0
    def check_proposal(tokens):
        nonlocal max_raw, max_probability
        raw = adapter.proposer.ensure(tokens)
        expected = model.forward_selected(tokens[None], [tokens.numel()-1])[0, -1]
        raw_error = float((raw-expected).abs().max())
        probability_error = float((probabilities(raw, model.config)-probabilities(expected, model.config)).abs().max())
        max_raw = max(max_raw, raw_error)
        max_probability = max(max_probability, probability_error)
        return probabilities(raw, model.config)
    q = check_proposal(prefix)
    draft = q.argmax(-1)
    verified = adapter.verifier.verify_logits(prefix, draft)
    check_proposal(torch.cat((prefix, draft)))
    # Build an independent native cache at the same prompt, then use exactly the
    # reference generator's token-at-a-time path for every target distribution.
    state, logits, _ = reference._prefill_generation_state(
        prefix[None], model.horizons+1, require_batched=True)
    untouched = deepcopy(state)
    expected = [logits[0]]
    active = torch.ones(1, dtype=torch.bool, device=prefix.device)
    for token in draft:
        logits = reference._decode_generation_state(state, token[None], active)
        reference._advance_decode_state(state, token[None], active)
        expected.append(logits[0])
    expected = torch.stack(expected)
    torch.testing.assert_close(verified, expected, atol=0, rtol=0)
    correction = (draft[0]+1) % model.config.eos_token_id
    corrected = torch.cat((prefix, correction[None]))
    check_proposal(corrected)
    recovered = adapter.verifier.ensure(corrected)
    corrected_logits = reference._decode_generation_state(untouched, correction[None], active)[0]
    torch.testing.assert_close(recovered, corrected_logits, atol=0, rtol=0)
    return dict(passed=True, verifier_backend='native_sequential',
                proposal_comparison='diagnostic_only',
                proposal_probability_validation='speculative decoder validates actual q; acceptance uses actual q and native p',
                maximum_proposal_logit_error=max_raw,
                maximum_proposal_probability_error=max_probability,
                maximum_reference_logit_error=float((verified-expected).abs().max()),
                native_reference_atol=0, native_reference_rtol=0)


@torch.inference_mode()
def evaluate(exp, checkpoint_path):
    """Measure pre/post quality and generation against the same fixed reference."""
    from scripts.mtp_common import validate_evaluation_inputs
    trained=json.loads((output_dir(exp)/'trained.json').read_text())
    validate_evaluation_inputs(trained)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('Production generation benchmarks require CUDA and the original SPS cached generator')
    model = load_mtp(exp.horizons, device, backend=MTP_INFERENCE_BACKEND).eval()
    # Keep the immutable target separate even when the restored draft is frozen
    # for inference: its backbone may contain previously fine-tuned weights.
    reference = load_reference(device).requires_grad_(False).eval()
    data = TokenSequences('val', MTP_PROMPT_TOKENS, exp.horizons)
    starts = validation_starts(len(data.tokens), MTP_PROMPT_TOKENS, MTP_EVAL_PROMPTS)
    prompts = [data[start][:MTP_PROMPT_TOKENS].to(device) for start in starts]
    if len(prompts) != MTP_EVAL_PROMPTS:
        raise RuntimeError('Insufficient held-out benchmark prompts')
    if MTP_PROMPT_TOKENS + MTP_NEW_TOKENS > model.config.block_size:
        raise RuntimeError('Generation benchmark exceeds source context capacity')
    report = dict(verifier_backend='native_sequential',
        limitations=['Native sequential verification preserves the source decoding distribution; it cannot measure the potential speed of parallel verification.',
                     'Native branch cache copies and accepted-prefix replay are included in generation timings.',
                     'Proposal cache/full numerical differences are diagnostic; acceptance uses the actual proposal probabilities and the exact native target.'],
        source=str(MTP_SOURCE), checkpoint=str(checkpoint_path), horizons=exp.horizons,
        device=torch.cuda.get_device_name(device), torch_version=torch.__version__,
        settings=dict(batch_size=1, prompts=len(prompts), prompt_tokens=MTP_PROMPT_TOKENS,
                      max_new_tokens=MTP_NEW_TOKENS, temperature=MTP_TEMPERATURE, seed=MTP_SEED,
                      inference_backend=MTP_INFERENCE_BACKEND, warmup='one complete workload per method/mode/phase'),
        baseline={}, before={}, after={})
    # Original SPS never changes: reuse its measured baseline for both phases.
    for mode in ('greedy', 'sample'):
        benchmark_reference(reference, prompts[0], mode=mode)
        report['baseline'][mode] = [benchmark_reference(reference, prefix,
            mode=mode, seed=MTP_SEED+i) for i, prefix in enumerate(prompts)]
    for phase in ('before', 'after'):
        if phase == 'after':
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if checkpoint.get('design_signature') != trained['design_signature']:
                raise RuntimeError('Adaptation checkpoint design signature mismatch')
            restore_draft(model, checkpoint)
        section = report[phase]
        section['horizon_metrics'] = horizon_metrics(model, device)
        section['cached_logits'] = check_cached_logits(model, prompts[0], reference=reference)
        section['generation'] = {}
        for mode in ('greedy', 'sample'):
            benchmark_speculative(model, prompts[0], mode=mode, reference=reference)
            rows = [benchmark_speculative(model, prefix, mode=mode, seed=MTP_SEED+i, reference=reference)
                    for i, prefix in enumerate(prompts)]
            baseline = report['baseline'][mode]
            for row, base in zip(rows, baseline):
                if mode == 'greedy':
                    row['full_token_equality'] = row['token_ids'] == base['token_ids']
            accepted = [sum(row['accepted_by_horizon'][j] for row in rows) for j in range(exp.horizons)]
            proposed = [sum(row['proposed_by_horizon'][j] for row in rows) for j in range(exp.horizons)]
            total = sum(row['total_seconds'] for row in rows)
            baseline_total = sum(row['total_seconds'] for row in baseline)
            section['generation'][mode] = dict(prompts=rows, accepted_by_horizon=accepted,
                proposed_by_horizon=proposed,
                acceptance_by_horizon=[a/p if p else None for a,p in zip(accepted, proposed)],
                accepted_tokens_per_round=sum(accepted)/sum(row['rounds'] for row in rows),
                total_seconds=total, baseline_total_seconds=baseline_total,
                tokens_per_second=sum(row['tokens'] for row in rows)/total,
                baseline_tokens_per_second=sum(row['tokens'] for row in baseline)/baseline_total,
                speedup=(sum(row['tokens'] for row in rows)/total)/(sum(row['tokens'] for row in baseline)/baseline_total),
                end_to_end_time_ratio=baseline_total/total,
                greedy_full_token_equality=all(row.get('full_token_equality', True) for row in rows) if mode == 'greedy' else None)
    destination = output_dir(exp)/'evaluation.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(destination)
    if not all(report[phase]['generation']['greedy']['greedy_full_token_equality'] for phase in ('before', 'after')):
        raise RuntimeError(f'Greedy speculative output differs from original SPS; see {destination}')
    return report


def main(argv=None):
    import argparse
    from pathlib import Path
    import experiments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--filter', required=True)
    args = parser.parse_args(argv)
    if experiments.SUITE not in {'mtp', 'mtp-unfreeze'}:
        raise RuntimeError('Select SPS_EXPERIMENT_SUITE=mtp or mtp-unfreeze')
    matches = experiments.collect_experiments(args.filter)
    if len(matches) != 1:
        raise ValueError('Select exactly one MTP experiment')
    exp = matches[0]
    trained = json.loads((output_dir(exp)/'trained.json').read_text())
    if trained['design_signature'] != experiments.design_signature():
        raise RuntimeError('Stale training identity')
    checkpoint = Path(trained['checkpoint'])
    if not checkpoint.is_file():
        raise RuntimeError('Missing final adaptation checkpoint')
    evaluate(exp, checkpoint)


if __name__ == '__main__':
    main()
