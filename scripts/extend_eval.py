"""One-GPU checkpoint/workload evaluation, independent of long sibling tasks."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from scripts import extend_common as sc, extend_config as c
from scripts.mlspace_launch import store_json


def evaluate(variant, exp, workload, limit=None):
    import torch
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from langdetect import DetectorFactory
    from lm_eval.tasks.ifeval import utils, instructions, instructions_registry, instructions_util
    from scripts import instruct_config as source
    from scripts.instruct_common import load_model, nll
    from scripts.instruct_data import rows
    from scripts.instruct_evaluate import ensure_ifeval_resources, evaluate_generation, likelihood
    sc.source_plan()
    checkpoint = sc.validate_checkpoint(variant,exp)
    root = sc.output_dir(variant,exp,workload)/('smoke' if limit else 'eval')
    root.mkdir(parents=True,exist_ok=True)
    gate = json.loads((sc.e.SANITY_ROOT/'generation-protocol.json').read_text())
    if gate['design_signature'] != c.SOURCE_SIGNATURE or gate['generation'] != source.GENERATION:
        raise RuntimeError('Source generation protocol mismatch')
    DetectorFactory.seed = source.SEED
    ensure_ifeval_resources()
    evaluator_code = {Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                      for m in (utils,instructions,instructions_registry,instructions_util)}
    original = json.loads((sc.e.output_dir(exp)/'eval/manifest.json').read_text())
    if (original['design_signature'] != c.SOURCE_SIGNATURE or original['ifeval_code'] != evaluator_code
            or original['lm_eval_version'] != importlib.metadata.version('lm_eval')):
        raise RuntimeError('Scoring implementation differs from source evaluation')
    manifest = dict(design_signature=sc.signature(),source_signature=c.SOURCE_SIGNATURE,
        experiment=exp.__dict__,checkpoint=checkpoint,workload=workload,limit=limit,
        generation=source.GENERATION,evaluator_code=evaluator_code,
        datasets={source.IFEVAL_DATASET:source.IFEVAL_REVISION,source.MMLU_DATASET:source.MMLU_REVISION})
    meta = root/'manifest.json'
    if meta.exists() and json.loads(meta.read_text()) != manifest:
        raise RuntimeError('Evaluation resume provenance mismatch')
    store_json(meta,manifest)
    model = load_model(exp.recipe,'cuda').eval().requires_grad_(False)
    state = torch.load(checkpoint['path'],map_location='cpu',weights_only=False)
    model.load_state_dict(state['model']); del state
    tokenizer = AutoTokenizer.from_pretrained(source.MODEL,revision=source.REVISION)
    result = dict(design_signature=sc.signature(),experiment=exp.name,checkpoint=checkpoint,workload=workload)
    started = time.monotonic()
    if workload == 'quality':
        data = rows('test')
        result['assistant_nll'] = nll(model,data[:limit] if limit else data,torch.device('cuda'))
        store_json(root/'heldout.json',result['assistant_nll'])
        likelihood(model,tokenizer,root,limit)
    else:
        task,mode = workload.split('-',1)
        if task == 'ifeval':
            data = list(load_dataset(source.IFEVAL_DATASET,revision=source.IFEVAL_REVISION,split='train'))
        else:
            data = list(load_dataset(source.MMLU_DATASET,'all',revision=source.MMLU_REVISION,split='test'))
        if limit: data = data[:limit]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            result['metrics'] = evaluate_generation(model,tokenizer,data,task,mode,source.GENERATION[mode],
                root,sc.signature(),source.SEED,started,c.WORKLOAD_HOURS[workload]-c.STARTUP_SECONDS/3600)
    store_json(root/'results.json',result)
    if not limit:
        store_json(sc.output_dir(variant,exp,workload)/'evaluated.json',result)
    print(json.dumps(result),flush=True)
