import torch
from torch import nn

from scripts import qwen4b_drafter as jobs
from scripts import qwen4b_drafter_train as training


def test_blocks_share_prefix_features_without_crossing_anchors():
    tokens = torch.arange(12)[None]
    anchors = (2, 5)
    widths = (4, 2)
    inputs, positions, slots, causal, labels = training.block_batch(
        tokens, anchors, widths, "causal",
    )
    _, _, _, bidirectional, _ = training.block_batch(tokens, anchors, widths, "bidirectional")

    assert inputs.tolist() == [[2, 2, 2, 2, 2, 5, 5, 5]]
    assert labels.tolist() == [[2, 3, 4, 5, 6, 5, 6, 7]]
    assert positions[0, 12:].tolist() == labels[0].tolist()
    assert slots.tolist() == [0, 1, 2, 3, 4, 0, 1, 2]
    for query, (anchor, first, width, slot) in enumerate([
        (2, 12, 4, slot) for slot in range(5)
    ] + [(5, 17, 2, slot) for slot in range(3)]):
        assert causal[0, 0, query].nonzero().flatten().tolist() == [
            *range(anchor), *range(first, first + slot + 1),
        ]
        assert bidirectional[0, 0, query].nonzero().flatten().tolist() == [
            *range(anchor), *range(first, first + width + 1),
        ]


class IdentityDraft(nn.Module):
    mask_token_id = 42
    target_layer_ids = [1, 9, 17, 25, 33]

    def forward(self, *, noise_embedding):
        return noise_embedding


def test_independent_slots_start_equal_and_receive_separate_gradients():
    embedding = torch.tensor([3., 5.])
    shared = training.BlockDrafter(IdentityDraft(), embedding, "shared")
    separate = training.BlockDrafter(IdentityDraft(), embedding, "per-position")
    source = torch.zeros(1, 5, 2)
    slots = torch.tensor([0, 1, 2, 0, 1])
    a = shared(noise_embedding=source, slots=slots)
    b = separate(noise_embedding=source, slots=slots)
    torch.testing.assert_close(a, b)
    assert b[0, 0].tolist() == [0., 0.]
    b.sum().backward()
    assert separate.mask_embeddings.grad.tolist() == [[2., 2.], [1., 1.]] + [[0., 0.]] * 6
    assert separate(noise_embedding=source)[0, 4].tolist() == [3., 5.]


def test_sampling_stays_inside_response_and_contains_both_lengths():
    row = {"loss_mask": [0, 0, 0] + [1] * 22 + [0]}
    anchors, widths = training.sample_blocks(row, 91)
    assert (anchors, widths) == training.sample_blocks(row, 91)
    assert sorted(widths) == [4] * 8 + [8] * 8
    for anchor, width in zip(anchors, widths):
        assert all(row["loss_mask"][anchor : anchor + width + 1])


def test_payload_stages_one_gpu_per_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "signature", lambda: "f" * 64)
    monkeypatch.setattr(jobs.evaluation, "signature", lambda: "e" * 64)
    assert len(jobs.names("train")) == 4
    for phase in ("probe", "train", "eval"):
        for name in jobs.names(phase):
            payload = jobs.payload(phase, name, tmp_path)
            assert payload["instance_type"] == "a100.1gpu"
            assert payload["processes_per_worker"] == 1
            assert f"--phase {phase} --worker --filter {name}" in payload["script"]


def test_speedup_uses_the_same_greedy_outputs_in_every_arm():
    def row(index, seconds, same=True):
        return dict(index=index, repeat=0, tokens=[0, 1, 2],
                    decode_seconds=seconds, greedy_match=same)

    samples = {
        "ar": [row(1, 2), row(2, 1), row(3, 1)],
        "causal": [row(1, 1), row(2, 0.25, False), row(3, 0.25)],
        "bidirectional": [row(1, 1.5), row(2, 0.25), row(3, 0.25, False)],
    }
    result = jobs.evaluation.matched_throughput(samples)
    assert result["requests"] == result["prompts"] == 1
    assert result["arms"]["ar"]["tokens_per_second"] == 1
    assert result["arms"]["causal"]["speedup_vs_ar"] == 2
