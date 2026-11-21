"""Train only the Qwen attention adapters and SPS prediction vector."""
import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import experiments as e
from scripts import ar_sps_common as common
from scripts import ar_sps_config as c
from scripts import qwen_adapter_run as cfg
from scripts.mlspace_launch import git, store_json


def rows(split):
    path = cfg.SOURCE/'data'/f'{split}.bin'
    if cfg.digest(path) != cfg.DATA_SHA[split]:
        raise RuntimeError(f'Unexpected {split} data')
    return np.memmap(path, mode='r', dtype=np.uint32).reshape(-1, c.CONTEXT+1)


def attach(model, arm):
    if cfg.digest(cfg.WHEEL) != cfg.WHEEL_SHA:
        raise RuntimeError('PEFT wheel mismatch')
    sys.path.insert(0, str(cfg.WHEEL))
    from peft import LoraConfig, inject_adapter_in_model
    model.set_stage(True)
    inject_adapter_in_model(LoraConfig(r=cfg.RANK, lora_alpha=cfg.ALPHA, lora_dropout=0.,
        target_modules=list(cfg.TARGETS), use_rslora=arm.rs, use_dora=arm.dora,
        bias='none'), model.backbone)
    params = {name: param for name, param in model.backbone.named_parameters() if param.requires_grad}
    if (len(params) != model.config.num_hidden_layers*len(cfg.TARGETS)*(2+int(arm.dora))
            or not all('.lora_' in name for name in params) or not model.prediction.requires_grad):
        raise RuntimeError('Unexpected trainable parameters')
    return list(params.values())


def initial(arm, device):
    model = common.load_model('staged_sps', device)
    parent_bytes = cfg.PARENT.read_bytes()
    if hashlib.sha256(parent_bytes).hexdigest() != cfg.PARENT_SHA:
        raise RuntimeError('Parent bytes changed before worker start')
    with io.BytesIO(parent_bytes) as snapshot:
        state = torch.load(snapshot, map_location='cpu', weights_only=True)
    del parent_bytes
    if (state['step'], state['stage'], state['source'], state['revision'], state['design_signature']) != (
            cfg.START, True, c.MODEL, c.REVISION, cfg.SIGNATURE):
        raise RuntimeError('Unexpected parent checkpoint')
    model.load_state_dict(state['model'])
    del state
    probe = torch.arange(8, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        before = model(probe).logits
        params = attach(model, arm)
        torch.testing.assert_close(model(probe).logits, before, atol=2e-4, rtol=2e-4)
    return model, params


def train(arm, commit):
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl')
    world = dist.get_world_size()
    if world != 4 or c.GLOBAL_BATCH % (world*c.MICRO_BATCH):
        raise RuntimeError('Unexpected training topology')
    rank = dist.get_rank()
    model, params = initial(arm, device)
    ddp = DistributedDataParallel(model.train(), device_ids=[local], broadcast_buffers=False)
    optim = torch.optim.AdamW([dict(params=params, lr=arm.lr),
                               dict(params=[model.prediction], lr=cfg.PRED_LR)],
                              betas=c.BETAS, weight_decay=c.WEIGHT_DECAY)
    data = rows('train')
    accumulation = c.GLOBAL_BATCH//(world*c.MICRO_BATCH)
    for step in range(cfg.START, cfg.END):
        optim.zero_grad(set_to_none=True)
        for micro in range(accumulation):
            offset = step*c.GLOBAL_BATCH + (micro*world+rank)*c.MICRO_BATCH
            batch = torch.tensor(data[offset:offset+c.MICRO_BATCH].astype('int64'), device=device)
            sync = contextlib.nullcontext() if micro+1 == accumulation else ddp.no_sync()
            with sync, torch.autocast('cuda', dtype=torch.bfloat16):
                loss = common.loss(ddp, batch)/accumulation
                loss.backward()
        torch.nn.utils.clip_grad_norm_([*params, model.prediction], c.CLIP, error_if_nonfinite=True)
        optim.step()
        if rank == 0 and (step+1) % 100 == 0:
            print(json.dumps(dict(step=step+1, last_micro_ce=loss.item()*accumulation)), flush=True)
    if rank == 0:
        score = float(np.mean(common.nll(model, rows('val'), device)))
        if not math.isfinite(score):
            raise RuntimeError('Nonfinite validation NLL')
        root = cfg.directory(arm)
        root.mkdir(parents=True, exist_ok=True)
        record = dict(arm=arm.name, commit=commit, step=cfg.END, val_nll=score)
        path = root/'final.pt'
        torch.save(dict(record=record, params={name: param.detach().cpu()
                   for name, param in model.named_parameters() if param.requires_grad}), path.with_suffix('.tmp'))
        path.with_suffix('.tmp').replace(path)
        store_json(root/'trained.json', dict(**record, sha256=cfg.digest(path)))
    dist.barrier()
    dist.destroy_process_group()


def evaluate(arm, commit):
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoTokenizer
    if torch.cuda.device_count() != 1 or cfg.winner(commit) != arm:
        raise RuntimeError('Evaluate only the selected winner on one GPU')
    root = cfg.directory(arm)
    marker = cfg.validated_final(arm, commit)
    adapter_bytes = (root/'final.pt').read_bytes()
    if hashlib.sha256(adapter_bytes).hexdigest() != marker['sha256']:
        raise RuntimeError('Adapter checkpoint changed')
    state = torch.load(io.BytesIO(adapter_bytes), map_location='cpu', weights_only=True)
    device = torch.device('cuda')
    model, _ = initial(arm, device)
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    if dict(**state['record'], sha256=marker['sha256']) != marker or set(state['params']) != trainable:
        raise RuntimeError('Adapter parameter identity mismatch')
    model.load_state_dict(state['params'], strict=False)
    model.eval()
    class Adapter(HFLM):
        def _model_call(self, inps, attn_mask=None, labels=None):
            if attn_mask is not None or labels is not None:
                raise ValueError('Only causal likelihood scoring is supported')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                return model(inps).logits
    adapter = Adapter(pretrained=model.backbone, tokenizer=AutoTokenizer.from_pretrained(c.MODEL, revision=c.REVISION),
                      backend='causal', batch_size=c.EVAL_BATCH, max_length=c.CONTEXT, device='cuda', logits_cache=False)
    losses = common.nll(model, rows('test'), device)
    store_json(root/'eval/heldout.json', dict(nll=float(np.mean(losses)), per_sequence=losses))
    metrics = {}
    for task in e.TASKS:
        result = evaluator.simple_evaluate(model=adapter, tasks=[task], num_fewshot=0, batch_size=c.EVAL_BATCH,
            log_samples=True, random_seed=c.SEED, numpy_random_seed=c.SEED, torch_random_seed=c.SEED, fewshot_random_seed=c.SEED)
        (root/'eval'/f'{task}.json').write_text(json.dumps(result, default=lambda value:
            value.tolist() if hasattr(value, 'tolist') else str(value)))
        metrics[task] = result['results'][task][c.PRIMARY_METRICS[task]]
    store_json(root/'evaluated.json', dict(**marker, heldout_nll=float(np.mean(losses)), metrics=metrics))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('train', 'eval'))
    parser.add_argument('arm', choices=[arm.name for arm in cfg.experiment_configs])
    args = parser.parse_args()
    if e.SUITE != cfg.SUITE:
        parser.error('Incorrect Qwen source suite')
    arm = next(arm for arm in cfg.experiment_configs if arm.name == args.arm)
    (train if args.phase == 'train' else evaluate)(arm, git('rev-parse', 'HEAD'))


if __name__ == '__main__':
    main()
