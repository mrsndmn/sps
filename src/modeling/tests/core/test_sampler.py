"""Distributed pilot data coverage and rank-independent initialization."""
import itertools
import random

import pytest

from training.sampler import FixedRandomChunkDistributedSampler, initial_sampler_offset
from experiments import BLOCK_SIZE, TRAIN_TOKENS, collect_experiments, overrides


def test_pilot_corpus_supplies_every_rank_for_the_entire_training_budget():
    # Reproduce the corpus size that exhausted rank three with the old defaults.
    dataset_len = 2231741898 - BLOCK_SIZE
    settings = dict(arg.split('=', 1) for arg in overrides(collect_experiments()[0]))
    samples_needed = TRAIN_TOKENS // BLOCK_SIZE // 4
    offset = initial_sampler_offset(dataset_len, 1337)
    for rank in range(4):
        sampler = FixedRandomChunkDistributedSampler(dataset_len, 4, rank, BLOCK_SIZE,
            chunk_size_units=int(settings['training.chunk_size_units']), start_offset=offset)
        assert len(sampler) >= samples_needed
        assert len(list(itertools.islice(sampler, 6))) == 6


def test_chunk_shards_are_disjoint_and_cover_sequence_starts():
    block = 16
    offset = 11
    units = 1025
    shards = [list(FixedRandomChunkDistributedSampler(offset + units * block, 4, rank,
        block, chunk_size_units=32, start_offset=offset)) for rank in range(4)]
    flattened = list(itertools.chain.from_iterable(shards))
    assert sorted(flattened) == list(range(offset, offset + units * block, block))
    assert len(set(flattened)) == len(flattened)


def test_initial_offset_is_independent_of_rank_rng_and_preserves_it():
    offsets = []
    for rank in range(4):
        random.seed(1337 + rank)
        state = random.getstate()
        offsets.append(initial_sampler_offset(2_000_000, 1337))
        assert random.getstate() == state
    assert len(set(offsets)) == 1


def test_resume_preserves_the_remaining_rank_sequence_order():
    kwargs = dict(dataset_len=10000, num_replicas=4, rank=2, block_size=16,
                  chunk_size_units=32, start_offset=7)
    full = list(FixedRandomChunkDistributedSampler(**kwargs))
    resumed = list(FixedRandomChunkDistributedSampler(**kwargs, resume_samples_seen_per_rank=13))
    assert resumed == full[13:]


def test_initial_offset_rejects_empty_data():
    with pytest.raises(ValueError, match='at least one sequence'):
        initial_sampler_offset(0, 1337)
