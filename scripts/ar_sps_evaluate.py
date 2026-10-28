"""Matched native/SPS lm-eval scoring and held-out per-sequence losses."""
import argparse
import json

import numpy as np
import torch

import experiments as e
from scripts import ar_sps_config as c
from scripts.ar_sps_common import load_model, restore, nll
from scripts.ar_sps_data import rows
from scripts.mlspace_launch import store_json


def audit():
    device = torch.device('cuda')
    model = load_model('original_ar', device).eval()
    batch = torch.tensor(rows('val')[:1].astype('int64'), device=device)[:, :32]
    with torch.no_grad():
        expected = model.backbone(batch, use_cache=False).logits
        actual = model(batch).logits
    torch.testing.assert_close(actual, expected, atol=c.PARITY_ATOL, rtol=c.PARITY_RTOL)
    captured = []
    hook = model.backbone.model.norm.register_forward_hook(lambda module, args, output: captured.append(output.detach()))
    with torch.no_grad():
        model(batch)
        model.sps = True
        full = model(batch).logits
        prefix = model(batch[:, :16]).logits
    hook.remove()
    torch.testing.assert_close(captured[1][:, ::2], captured[0], atol=c.PARITY_ATOL, rtol=c.PARITY_RTOL)
    torch.testing.assert_close(prefix, full[:, :16], atol=c.PARITY_ATOL, rtol=c.PARITY_RTOL)
    store_json(e.SANITY_ROOT/'native-parity.json', dict(max_abs=(actual-expected).abs().max().item(),
                                                      design_signature=e.design_signature()))


def evaluate(exp, checkpoint=None, label='source'):
    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator
    from transformers import AutoTokenizer
    device = torch.device('cuda')
    model = load_model(exp.recipe, device).eval()
    metadata = restore(checkpoint, model, device) if checkpoint else None
    tokenizer = AutoTokenizer.from_pretrained(c.MODEL, revision=c.REVISION)

    class Adapter(HFLM):
        def _model_call(self, inps, attn_mask=None, labels=None):
            if attn_mask is not None or labels is not None:
                raise ValueError('Only causal likelihood scoring is supported')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                return model(inps).logits

    adapter = Adapter(pretrained=model.backbone, tokenizer=tokenizer, backend='causal',
                      batch_size=c.EVAL_BATCH, max_length=c.CONTEXT, device='cuda',
                      logits_cache=False)
    destination = e.output_dir(exp)/'eval'/label
    destination.mkdir(parents=True, exist_ok=True)
    losses = nll(model, rows('test'), device)
    store_json(destination/'heldout.json', dict(nll=float(np.mean(losses)), per_sequence=losses,
               step=metadata['step'] if metadata else 0, design_signature=e.design_signature()))
    # Task-scoped outputs survive a later timeout and retain per-example pairing.
    for task in e.TASKS:
        result = evaluator.simple_evaluate(model=adapter, tasks=[task], num_fewshot=0,
                                          batch_size=c.EVAL_BATCH, log_samples=True,
                                          random_seed=c.SEED, numpy_random_seed=c.SEED,
                                          torch_random_seed=c.SEED, fewshot_random_seed=c.SEED)
        (destination/f'{task}.json').write_text(json.dumps(result, default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)))
    store_json(destination/'complete.json', dict(design_signature=e.design_signature()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter')
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    if args.audit:
        return audit()
    matches = e.collect_experiments(args.filter)
    if len(matches) != 1:
        raise ValueError('Select one evaluation arm')
    exp = matches[0]
    if not exp.trainable:
        evaluate(exp)
    else:
        labels = ['stage_endpoint', 'final', 'best']
        if exp.recipe == 'staged_sps':
            labels.append('embedding_best')
        for label in labels:
            evaluate(exp, e.output_dir(exp)/f'{label}.pt', label)


if __name__ == '__main__':
    main()
