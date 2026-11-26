import json

import pytest

from data.pilot_data import checked_manifest, prepare_memmaps
from experiments import paper_reproduction as study
from experiments.pilot import DATASET_REPO, DATASET_REVISION, VAL_FRACTION, VAL_SEED
from orchestration.named.registry import family_runs


def test_catalog_matches_training_recipes():
    from experiments.paper import recipes

    catalog = family_runs(study.PaperSPS.family)
    assert set(catalog) == {("data", "prepare"), ("s", "train"), ("m", "train"), ("xl", "train"),
                            *((f"part-{i}", "prepare") for i in range(1,5))}
    configs = study.PaperSPS.experiment_configs()
    assert isinstance(configs, list)
    assert [(exp.name, exp.phase) for exp in configs] == list(catalog)
    corpus, *rest = configs
    training = [r for r in rest if isinstance(r, study.TrainSPS)]
    assert tuple(f for part in corpus.parts for f in part.files) == corpus.extra_files
    assert isinstance(corpus, study.PrepareFineWeb)
    assert len(corpus.files) == len(set(corpus.files)) == 36
    for exp in training:
        assert exp.corpus is corpus
        recipe = recipes[exp.recipe.name]
        assert recipe == exp.recipe
        assert (recipe.window_size, recipe.max_tokens, recipe.block_size) == (0, 20_000_000_000, 4096)


def test_reuse_prefix_and_reject_truncated_memmap(tmp_path):
    pytest.importorskip("tiktoken")
    pytest.importorskip("pyarrow")
    pytest.importorskip("omegaconf")
    from unittest.mock import patch

    src, dest = tmp_path / "prefix", tmp_path / "corpus"
    files = ("part.parquet",)
    src.mkdir()
    counts = {"train": 3, "val": 1048576}
    for split, count in counts.items():
        (src / f"{split}.bin").write_bytes(b"\x01\x00" * count)
    manifest = dict(repo=DATASET_REPO, revision=DATASET_REVISION, tokenizer="gpt2",
                    dtype="uint16", files=list(files), tokens=counts, sources=[],
                    val_fraction=VAL_FRACTION, val_seed=VAL_SEED)
    (src / "manifest.json").write_text(json.dumps(manifest))
    with patch("tiktoken.get_encoding"):
        prepare_memmaps(dest, files, 0, 3, prefix_dir=src, prefix_files=files)
    assert checked_manifest(dest, files)["tokens"] == counts
    assert (dest / "train.bin").read_bytes() == (src / "train.bin").read_bytes()
    (dest / "train.bin").write_bytes(b"")
    with pytest.raises(RuntimeError, match="Incomplete train"):
        checked_manifest(dest, files)


@pytest.mark.parametrize("args", [["--worker"], ["--worker", "--dry"]])
def test_direct_worker_cannot_write_published_data(monkeypatch, args):
    import sys
    from orchestration.paper_reproduction import main

    monkeypatch.delenv("SPS_JOB_OUTPUT", raising=False)
    monkeypatch.setattr(sys, "argv", ["paper_reproduction", "data", *args])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_merge_orders_parts_and_checks_each_split(tmp_path):
    from data.pilot_data import prepare_memmaps
    parts = []
    expected = {"train": b"", "val": b""}
    for index in (1, 2):
        dest = tmp_path / str(index)
        dest.mkdir()
        files = (f"{index}.parquet",)
        counts = {"train": index, "val": 524288 + index}
        for split, count in counts.items():
            data = index.to_bytes(2, "little") * count
            (dest / f"{split}.bin").write_bytes(data)
            expected[split] += data
        manifest = dict(repo=DATASET_REPO, revision=DATASET_REVISION, tokenizer="gpt2",
                        dtype="uint16", files=list(files), tokens=counts, sources=[],
                        val_fraction=VAL_FRACTION, val_seed=VAL_SEED)
        (dest / "manifest.json").write_text(json.dumps(manifest))
        parts.append((dest, files))
    files = ("1.parquet", "2.parquet")
    output = tmp_path / "combined"
    result = prepare_memmaps(output, files, 0, 3, parts=parts)
    for split, data in expected.items():
        assert (output / f"{split}.bin").read_bytes() == data
        assert result["tokens"][split] == len(data) // 2
    with pytest.raises(ValueError, match="ordered shard"):
        prepare_memmaps(tmp_path / "reordered", files, 0, 3, parts=parts[::-1])
    (parts[1][0] / "val.bin").write_bytes(b"")
    with pytest.raises(RuntimeError, match="Incomplete val"):
        prepare_memmaps(tmp_path / "partial", files, 0, 3, parts=parts)


def test_merge_retry_removes_restored_private_prefix(tmp_path, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    import sys
    from orchestration import paper_reproduction as worker
    from data import pilot_data

    shared = tmp_path / "pilot"
    shared.mkdir()
    (shared / "manifest.json").touch()
    corpus = replace(study.PaperSPS.experiment_configs()[0], data_root=tmp_path, prefix=shared)
    training = replace(study.TrainSPS.experiment_configs()[-1], corpus=corpus)
    work = tmp_path / "reserved"
    (work / "prefix").mkdir(parents=True)
    def merge(*args, **kwargs):
        assert kwargs["prefix_dir"] == shared

    monkeypatch.setattr(study.PaperSPS, "experiment_configs", classmethod(lambda cls: [corpus, training]))
    monkeypatch.setattr(worker, "RunDirectory", lambda *args: SimpleNamespace(work=work))
    monkeypatch.setattr(worker, "check_data", lambda *args: {})
    monkeypatch.setattr(worker, "tokenized_parts", lambda *args: [])
    monkeypatch.setattr(pilot_data, "prepare_memmaps", merge)
    monkeypatch.setenv("SPS_JOB_OUTPUT", str(corpus.output))
    monkeypatch.setenv("SPS_JOB_WORK", str(work))
    monkeypatch.setattr(sys, "argv", ["paper_reproduction", "data", "--worker"])
    worker.main()
    assert not (work / "prefix").exists()
    assert (shared / "manifest.json").exists()
