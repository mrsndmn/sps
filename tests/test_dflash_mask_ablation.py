import pytest
import torch

from workloads.distillation.dflash_mask_ablation import (
    Experiment,
    block_attention,
    commit_round,
    context_features,
    round_is_complete,
    summary,
)


@pytest.mark.parametrize("prefix", [0, 7])
def test_block_visibility_keeps_all_prefix_keys_and_changes_only_block(prefix):
    width = 5
    positions = torch.arange(prefix + width + 1)
    control = block_attention(positions, prefix, width, "bidirectional")[0, 0]
    sps = block_attention(positions, prefix, width, "causal")[0, 0]
    assert control.shape == sps.shape == (width, prefix + width)
    assert control.all()
    assert sps[:, :prefix].all()
    assert torch.equal(
        sps[:, prefix:], torch.ones(width, width, dtype=torch.bool).tril()
    )


def test_commit_contiguous_prefix_and_target_bonus_with_terminal_trim():
    assert commit_round([1, 2, 3], [1, 9, 3, 4], 5, {8}) == ([1, 9], 1, False)
    assert commit_round([1, 2, 3], [1, 2, 3, 4], 5, {8}) == ([1, 2, 3, 4], 3, False)
    assert commit_round([1, 8, 3], [1, 8, 3, 4], 5, {8}) == ([1, 8], 3, True)
    assert commit_round([1, 2, 3], [1, 2, 3, 4], 2, {8}) == ([1, 2], 3, True)


def test_keep_exact_cap_or_bonus_eos_and_exclude_truncated_rounds():
    args = ([1, 2, 3], [1, 2, 3, 4])
    for remaining, stop, expected in (
        (4, {8}, True),
        (5, {4}, True),
        (5, {2}, False),
        (2, {8}, False),
    ):
        suffix, count, finished = commit_round(*args, remaining, stop)
        assert finished
        assert round_is_complete(suffix, count) == expected


def test_context_uses_output_indices_and_excludes_rejected_positions():
    from types import SimpleNamespace

    hidden = [torch.full((1, 4, 1), i) for i in range(5)]
    projected_input = context_features(SimpleNamespace(hidden_states=hidden), [1, 3], 2)
    assert projected_input.tolist() == [[[2, 4], [2, 4]]]


def test_report_excludes_incomplete_rounds_from_acceptance_and_first_token_from_speed():
    sample = dict(
        tokens=[1, 2, 3, 4, 5],
        decode_seconds=2,
        acceptance=[2, 0],
        verify_rounds=3,
        draft_gpu_ms=9,
        verify_gpu_ms=12,
        greedy_match=True,
    )
    result = summary([sample], Experiment("causal", 4))
    assert result["tokens_per_second"] == 2
    assert result["tau"] == 2
    assert result["acceptance"] == 0.25
    assert result["acceptance_by_position"] == [0.5, 0.5, 0, 0]
    assert result["draft_gpu_ms_per_round"] == 3
    assert result["verify_gpu_ms_per_round"] == 4
