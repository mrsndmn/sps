"""Shared SPS/adaptation loading, document-safe targets and data order."""
from pathlib import Path
import json
import experiments as e
import hashlib
import numpy as np
import torch

from experiments import MTP_SOURCE, MTP_DATA, MTP_MAX_HORIZONS, MTP_SEED


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def warmstart_checkpoint(horizons):
    return e.MTP_PARENT_ROOT / f's_sps_mtp_n{horizons}_emb_1b' / 'adaptation_final.pt'


def warmstart_metadata(horizons, world_size=4):
    """Validate the completed frozen parent, independent of the new design hash."""
    path = warmstart_checkpoint(horizons)
    trained = json.loads((path.parent / 'trained.json').read_text())
    if trained.get('checkpoint') != str(path):
        raise RuntimeError('Parent trained marker checkpoint mismatch')
    validate_evaluation_inputs(trained)
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('source') != str(MTP_SOURCE) or state.get('horizons') != horizons:
        raise RuntimeError('Parent adaptation source/horizon mismatch')
    signature = trained.get('design_signature')
    if not signature or state.get('design_signature') != signature:
        raise RuntimeError('Parent checkpoint design signature mismatch')
    tokens, step = state.get('tokens'), state.get('step')
    if tokens != e.MTP_PARENT_INPUT_TOKENS or not isinstance(step, int) or step <= 0:
        raise RuntimeError('Parent checkpoint is not the completed input budget')
    if tokens != step * e.MTP_PARENT_STEP_TOKENS:
        raise RuntimeError('Parent checkpoint step/token mismatch')
    source = torch.load(MTP_SOURCE, map_location='cpu', weights_only=False, mmap=True)
    width = source['model_args']['hidden_size']
    del source
    embeddings = state.get('new_embeddings')
    if (not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2
            or embeddings.shape != (horizons - 1, width)
            or not torch.isfinite(embeddings).all()):
        raise RuntimeError('Parent adaptation embedding shape/value mismatch')
    if state.get('train_backbone') or 'backbone' in state:
        raise RuntimeError('Warmstart must be an embedding-only parent')
    per_rank_tokens = e.MTP_PARENT_CONTEXT * world_size
    if world_size <= 0 or tokens % per_rank_tokens:
        raise RuntimeError('Parent input budget cannot map to exact per-rank samples')
    return dict(path=str(path), sha256=file_sha256(path), parent_tokens=tokens,
        parent_step=step, parent_samples_per_rank=tokens // per_rank_tokens,
        source_sha256=trained['source_sha256'],
        data_manifest_sha256=trained['data_manifest_sha256'],
        parent_design_signature=signature)


@torch.no_grad()
def restore_draft(model, state):
    """Restore every draft weight while retaining the model's tied parameters."""
    if state.get('horizons') != model.horizons or state.get('source') != str(MTP_SOURCE):
        raise RuntimeError('Adaptation checkpoint source/horizon mismatch')
    embeddings = state.get('new_embeddings')
    if not isinstance(embeddings, torch.Tensor) or embeddings.shape != model.new_embeddings.shape:
        raise RuntimeError('Adaptation checkpoint embedding shape mismatch')
    if state.get('train_backbone') and 'backbone' not in state:
        raise RuntimeError('Full adaptation checkpoint is missing backbone weights')
    if 'backbone' in state:
        backbone = state['backbone']
        if not torch.equal(backbone['transformer.wte.weight'], backbone['lm_head.weight']):
            raise RuntimeError('Adaptation checkpoint tied weights disagree')
        model.reference.load_state_dict(backbone)
        model.has_adapted_backbone = True
        if model.reference.transformer.wte.weight is not model.reference.lm_head.weight:
            raise RuntimeError('Draft input/output weights lost their tied identity')
    model.new_embeddings.copy_(embeddings)


def load_reference(device):
    """Load a fresh original SPS model; never share storage with a draft."""
    from modeling.models.sps import SPSConfig, SPSModel
    ckpt = torch.load(MTP_SOURCE, map_location='cpu', weights_only=False)
    reference = SPSModel(SPSConfig(**ckpt['model_args']))
    reference.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in ckpt['model'].items()})
    return reference.to(device)


def load_mtp(horizons, device, checkpoint_path=None, backend='auto', train_backbone=False):
    from modeling.models.sps.mtp import SPSMTP
    if checkpoint_path is None and getattr(e, 'MTP_UNFREEZE', False):
        checkpoint_path = Path(warmstart_metadata(horizons)['path'])
    reference = load_reference(device)
    model = SPSMTP(reference, horizons, backend=backend, train_backbone=train_backbone).to(device)
    if checkpoint_path is not None:
        adaptation = torch.load(checkpoint_path, map_location=device, weights_only=False)
        restore_draft(model, adaptation)
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


def validate_evaluation_inputs(trained):
    """Bind downstream results to the actual frozen model and held-out corpus."""
    for path, key, label in (
        (MTP_SOURCE, 'source_sha256', 'source checkpoint'),
        (MTP_DATA/'manifest.json', 'data_manifest_sha256', 'data manifest'),
    ):
        expected=trained.get(key)
        digest=hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda:handle.read(8*1024*1024),b''):
                digest.update(chunk)
        if not expected or digest.hexdigest()!=expected:
            raise RuntimeError(f'Evaluation {label} changed or has no recorded identity')
