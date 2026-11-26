import torch
from torch import nn

from workloads.distillation import qwen4b_drafter as jobs, qwen4b_drafter_data as prepared
from workloads.distillation import qwen4b_drafter_train as training


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
    assert len(jobs.names("train")) == 4
    for dataset in training.DATASETS:
        for phase in ("probe", "train", "eval"):
            for name in jobs.names(phase, dataset):
                payload = jobs.payload(phase, name, tmp_path, dataset)
                assert payload["instance_type"] == "a100.1gpu"
                assert payload["processes_per_worker"] == 1
                assert f"--phase {phase} --dataset {dataset} --worker --filter {name}" in payload["script"]


def test_prepared_selection_uses_context_and_holds_out_whole_prompts():
    class Tokenizer:
        def decode(self, ids, *, skip_special_tokens):
            return str(ids[0])

    class Overlap:
        def matches(self, text):
            return text == "8"

    def row(prompt, response_length):
        ids = [prompt] + [66] * response_length
        return {"input_ids": ids, "loss_mask": [0] + [1] * response_length, "seq_len": len(ids)}

    rows = [row(prompt, 128) for prompt in (1, 1, 2, 2, 3, 4, 8)]
    rows += [row(5, 256), row(6, 127)]
    scattered = row(7, 255)
    scattered["loss_mask"] = [0, 1] * 128
    rows.append(scattered)
    args = dict(context=256, horizon=4, validation=2, accumulation=1, seed=1337)
    result = prepared.select_rows(rows, Tokenizer(), Overlap(), **args)
    assert result == prepared.select_rows(rows, Tokenizer(), Overlap(), **args)
    train_prompts = {rows[index]["input_ids"][0] for index in result["train"]}
    val_prompts = {rows[index]["input_ids"][0] for index in result["val"]}
    assert len(val_prompts) == 2
    assert not train_prompts & val_prompts
    assert all(sum(rows[index]["loss_mask"]) == 128 for index in result["train"] + result["val"])
    assert result["rejected"]["length"] == 1
    assert result["rejected"]["min_loss_tokens"] == 1
    assert result["rejected"]["anchors"] == 1
    assert result["rejected"]["overlap"] == 1


def test_prepared_arrow_loader_handles_newer_hf_list_metadata(monkeypatch, tmp_path):
    import huggingface_hub
    import pyarrow as pa

    fields = {"input_ids": [[1, 2, 3]], "loss_mask": [[0, 1, 1]], "seq_len": [3]}
    table = pa.table(fields).replace_schema_metadata({b"huggingface": b"incompatible"})
    with pa.ipc.new_stream(tmp_path / "data.arrow", table.schema) as writer:
        writer.write_table(table)
    (tmp_path / "state.json").write_text('{"_data_files":[{"filename":"data.arrow"}]}')
    monkeypatch.setattr(prepared, "SOURCE_ROWS", 1)

    def snapshot(repo, **kwargs):
        assert repo == prepared.REPO
        assert kwargs["revision"] == prepared.REVISION and kwargs["local_files_only"]
        assert kwargs["cache_dir"] == prepared.HF_HOME / "hub"
        return tmp_path

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    rows = prepared.open_dataset()
    assert rows[0] == {key: values[0] for key, values in fields.items()}


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


def test_pairwise_drafter_rates_include_equal_outputs_that_diverge_from_ar():
    def row(tokens, seconds):
        return dict(index=1, repeat=0, tokens=tokens, decode_seconds=seconds, greedy_match=False)

    left = [row([1, 3], 2.)]
    right = [row([1, 3], 1.)]
    result = jobs.evaluation.compare_drafts(left, right)
    assert result["requests"] == result["prompts"] == 1
    assert result["first_tokens_per_second"] == 0.5
    assert result["second_over_first"] == 2
    assert jobs.evaluation.compare_drafts(left, [row([1, 4], 1.)])["requests"] == 0
