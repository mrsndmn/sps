"""Actual sampler capacity checks; requires the training environment's torch."""
import pytest

pytest.importorskip('torch')

from scripts import pilot_run
from experiments import TRAIN_TOKENS


def test_old_corpus_cannot_be_cycled_to_claim_a_tenfold_data_budget(monkeypatch):
    monkeypatch.setattr(pilot_run, 'TRAIN_TOKENS', TRAIN_TOKENS * 10)
    with pytest.raises(RuntimeError, match='Insufficient unique training data'):
        pilot_run.validate_training_coverage(dict(tokens=dict(train=2231741898)))


def test_larger_corpus_supplies_full_budget_on_every_rank(monkeypatch):
    monkeypatch.setattr(pilot_run, 'TRAIN_TOKENS', TRAIN_TOKENS * 10)
    coverage = pilot_run.validate_training_coverage(dict(tokens=dict(train=11500000000)))
    for row in coverage.values():
        assert len(row['available_per_rank']) == 4
        assert min(row['available_per_rank']) >= row['required_per_rank']
