"""Shared frozen-checkpoint loading, document-safe targets and data order."""
from pathlib import Path
import numpy as np
import torch

from experiments import MTP_SOURCE, MTP_DATA, MTP_MAX_HORIZONS, MTP_SEED


def load_mtp(horizons, device, checkpoint_path=None, backend='auto'):
    from modeling.models.sps import SPSConfig, SPSModel
    from modeling.models.sps.mtp import SPSMTP
    ckpt = torch.load(MTP_SOURCE, map_location='cpu', weights_only=False)
    reference = SPSModel(SPSConfig(**ckpt['model_args']))
    reference.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in ckpt['model'].items()})
    del ckpt
    model = SPSMTP(reference, horizons, backend=backend).to(device)
    if checkpoint_path is not None:
        adaptation = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if adaptation['horizons'] != horizons or adaptation['source'] != str(MTP_SOURCE):
            raise RuntimeError('Adaptation checkpoint source/horizon mismatch')
        with torch.no_grad():
            model.new_embeddings.copy_(adaptation['new_embeddings'])
    return model


def shifted_targets(tokens, context, horizons, *, eos, pad):
    """Do not supervise a target beyond any EOS/padding in its visible gap."""
    if tokens.shape[1] < context + horizons:
        raise ValueError('Need the entire requested future horizon')
    targets = []
    valid = (tokens[:, :context] != pad)
    for j in range(1, horizons + 1):
        previous = tokens[:, j-1:j-1+context]
        target = tokens[:, j:j+context]
        valid = valid & (previous != eos) & (previous != pad) & (target != pad)
        targets.append(torch.where(valid, target, -100))
    return torch.stack(targets, dim=-1)


def sequence_starts(token_count, context, world, rank, *, seed=MTP_SEED, resume=0):
    from training.sampler import FixedRandomChunkDistributedSampler, initial_sampler_offset
    length = token_count - context - MTP_MAX_HORIZONS + 1
    return FixedRandomChunkDistributedSampler(length, world, rank, context,
        chunk_size_units=1024, seed=seed, start_offset=initial_sampler_offset(length,seed),
        resume_samples_seen_per_rank=resume)


class TokenSequences(torch.utils.data.Dataset):
    def __init__(self, split, context, horizons=MTP_MAX_HORIZONS):
        self.tokens=np.memmap(MTP_DATA/f'{split}.bin',dtype=np.uint16,mode='r')
        self.context=context
        self.horizons=horizons
    def __len__(self):
        return len(self.tokens)-self.context-self.horizons+1
    def __getitem__(self, start):
        return torch.from_numpy(np.array(self.tokens[start:start+self.context+self.horizons],dtype=np.int64))


def validation_starts(token_count, context, count):
    # Match prefixes across every N; reserve the largest future horizon.
    return [i*context for i in range(min(count,(token_count-MTP_MAX_HORIZONS)//context))]
