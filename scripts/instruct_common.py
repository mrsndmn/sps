"""Assistant-only objectives and checkpoint provenance for instruct adaptation."""
import json
import math

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def configuration():
    from scripts import instruct_config
    return instruct_config


def load_model(recipe, device):
    from transformers import AutoModelForCausalLM
    from modeling.models.sps.pretrained import PretrainedSPS
    c = configuration()
    torch.manual_seed(c.SEED)
    base = AutoModelForCausalLM.from_pretrained(c.MODEL, revision=c.REVISION,
                                              torch_dtype=torch.float32, attn_implementation='sdpa')
    base.config.use_cache = False
    return PretrainedSPS(base, sps=recipe not in {'original_ar', 'continued_ar'}).to(device)


def load_teacher(device):
    model = load_model('original_ar', device)
    model.requires_grad_(False)
    return model.eval()


def masked_loss(logits, teacher_logits, targets, target_mask, *, weight=0., temperature=1., chunk_size=64):
    """Return objective SUM, detached CE/KL sums, and target-token count.

    Inputs are already next-token aligned. Chunk checkpointing prevents retaining
    full-vocabulary FP32 softmax intermediates across the whole sequence.
    """
    if not 0 <= weight <= 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Invalid distillation parameters')
    if logits.shape[:-1] != targets.shape or targets.shape != target_mask.shape:
        raise ValueError('Targets and mask must align with logits')
    if not torch.all((target_mask == 0) | (target_mask == 1)):
        raise ValueError('Loss mask must be binary')
    if chunk_size < 1:
        raise ValueError('Chunk size must be positive')
    if weight and (teacher_logits is None or teacher_logits.shape != logits.shape):
        raise ValueError('KD requires aligned frozen teacher logits')
    flat = logits.flatten(0, -2)
    gold = targets.flatten()
    selected = target_mask.flatten().bool().nonzero(as_tuple=False).flatten()
    if not len(selected):
        raise ValueError('No supervised target tokens')
    teacher_flat = teacher_logits.detach().flatten(0, -2) if weight else None
    total = logits.new_zeros((), dtype=torch.float32)
    ce_total = total.detach().clone(); kd_total = total.detach().clone()

    def chunk_objective(student, teacher, labels):
        ce = F.cross_entropy(student.float(), labels, reduction='sum')
        kd = ce.new_zeros(())
        if weight:
            log_s = F.log_softmax(student.float() / temperature, dim=-1)
            log_t = F.log_softmax(teacher.float() / temperature, dim=-1)
            kd = F.kl_div(log_s, log_t, reduction='sum', log_target=True) * temperature**2
        return (1-weight)*ce + weight*kd, ce.detach(), kd.detach()

    for offset in range(0, len(selected), chunk_size):
        ids = selected[offset:offset+chunk_size]
        student = flat[ids]
        teacher = teacher_flat[ids] if weight else student.detach().new_empty(0)
        # Non-reentrant checkpoint supports frozen input and trainable prediction vectors.
        result = checkpoint(chunk_objective, student, teacher, gold[ids], use_reentrant=False) if torch.is_grad_enabled() and student.requires_grad else chunk_objective(student, teacher, gold[ids])
        value, ce, kd = result
        total = total + value
        ce_total += ce; kd_total += kd
    return total, ce_total, kd_total, len(selected)


def loss(model, row, device, *, teacher=None, weight=0., temperature=1.):
    tokens = torch.tensor(row['input_ids'], dtype=torch.long, device=device)[None]
    mask = torch.tensor(row['loss_mask'], dtype=torch.bool, device=device)[None, 1:]
    logits = model(tokens[:, :-1]).logits
    teacher_logits = None
    if weight:
        if teacher is None:
            raise ValueError('Missing teacher')
        with torch.no_grad():
            teacher_logits = teacher(tokens[:, :-1]).logits
    return masked_loss(logits, teacher_logits, tokens[:, 1:], mask,
                       weight=weight, temperature=temperature)


@torch.no_grad()
def nll(model, data, device):
    was_training = model.training
    model.eval()
    examples = []; modes = {}; summed = 0.; count = 0
    try:
        for row in data:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                _, ce, _, n = loss(model, row, device)
            value = ce.item(); summed += value; count += n
            mode = modes.setdefault(row['mode'], {'loss_sum': 0., 'tokens': 0})
            mode['loss_sum'] += value; mode['tokens'] += n
            examples.append(dict(source_id=row['source_id'], mode=row['mode'], nll=value/n, tokens=n))
    finally:
        model.train(was_training)
    if not count:
        raise ValueError('Empty evaluation set')
    return dict(nll=summed/count, tokens=count, by_mode={k: dict(v, nll=v['loss_sum']/v['tokens']) for k,v in modes.items()}, per_example=examples)


def save(path, model, optimizer, step, best, stage, plan):
    import experiments as e
    c = configuration()
    temporary = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                    best=best, stage=stage, plan=plan, design_signature=e.design_signature(),
                    source=c.MODEL, revision=c.REVISION), temporary)
    temporary.replace(path)


def restore(path, model, device):
    import experiments as e
    c = configuration()
    state = torch.load(path, map_location='cpu', weights_only=False)
    if (state['design_signature'] != e.design_signature() or state['source'] != c.MODEL
            or state['revision'] != c.REVISION):
        raise RuntimeError('Checkpoint provenance mismatch')
    model.load_state_dict(state['model'])
    return state


def read_plan():
    import experiments as e
    plan = json.loads((e.ARTIFACTS_ROOT/'prepared.json').read_text())
    if plan['design_signature'] != e.design_signature():
        raise RuntimeError('Stale preparation plan')
    return plan
