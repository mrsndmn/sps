"""Shared source loading, immutable identities and quality measurements."""
import json
import torch
from torch.nn import functional as F

import experiments as e
from scripts import ar_sps_config as c
from modeling.models.sps.pretrained import PretrainedSPS


def load_model(recipe, device):
    from transformers import AutoModelForCausalLM
    torch.manual_seed(c.SEED)
    base = AutoModelForCausalLM.from_pretrained(c.MODEL, revision=c.REVISION,
                                              torch_dtype=torch.float32, attn_implementation='sdpa')
    return PretrainedSPS(base, sps=recipe not in {'original_ar', 'continued_ar'}).to(device)


def loss(model, batch):
    logits = model(batch[:, :-1]).logits
    return F.cross_entropy(logits.float().flatten(0, 1), batch[:, 1:].flatten())


@torch.no_grad()
def nll(model, data, device):
    was_training = model.training
    model.eval()
    values = []
    for start in range(0, len(data), c.EVAL_BATCH):
        batch = torch.tensor(data[start:start+c.EVAL_BATCH].astype('int64'), device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = model(batch[:, :-1]).logits.float()
            item = F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten(), reduction='none')
        values.extend(item.reshape(len(batch), -1).mean(-1).cpu().tolist())
    model.train(was_training)
    return values


def save(path, model, optimizer, step, best, stage, plan):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                    best=best, stage=stage, plan=plan, design_signature=e.design_signature(),
                    source=c.MODEL, revision=c.REVISION), temporary)
    temporary.replace(path)


def restore(path, model, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if (state['design_signature'] != e.design_signature() or state['source'] != c.MODEL
            or state['revision'] != c.REVISION):
        raise RuntimeError('Checkpoint provenance mismatch')
    model.load_state_dict(state['model'])
    return state


def read_plan():
    plan = json.loads((e.ARTIFACTS_ROOT/'prepared.json').read_text())
    if plan['design_signature'] != e.design_signature():
        raise RuntimeError('Stale preparation plan')
    return plan
