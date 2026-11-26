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
