"""Paired zero-shot chat generation on the full GSM8K test split."""
from orchestration.named.publication import work_path
import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import shlex

from orchestration.source_compatibility import source_hash
from experiments import instruct_config as source
from orchestration import mlspace_launch as launch
from workloads.readout import real_readout_eval as quality
from workloads.readout import real_readout_run as train
from workloads.serving import vllm_sps_common as common
from experiments import vllm_sps_config as c
from workloads.evaluation.mmlu_results import recover_samples
from workloads.readout.readout_archive import compatible_identity
from experiments.real_readout_gsm8k import (
    ROOT,
    HOURS,
    INSTRUCTION,
    experiment_configs,
)


NUMBER = re.compile(r"####[ \t]+([+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?)")


def answer(text):
    lines = text.rstrip().splitlines()
    match = NUMBER.fullmatch(lines[-1].strip()) if lines else None
    if match:
        value = Decimal(match[1].replace(",", ""))
        if not value:
            return "0"
        fixed = format(value, "f")
        return fixed.rstrip("0").rstrip(".") if "." in fixed else fixed
    return None


def dataset():
    from huggingface_hub import hf_hub_download
    from pyarrow.parquet import read_table
    from experiments.environment import HF_HOME
    from workloads.distillation.dflash_repro_eval import GSM8K, GSM8K_REVISION, GSM8K_FILE
    path = hf_hub_download(GSM8K, GSM8K_FILE, repo_type="dataset", revision=GSM8K_REVISION,
                           cache_dir=HF_HOME / "hub", local_files_only=True)
    docs = read_table(path, columns=["question", "answer"]).to_pylist()
    if len(docs) != 1319 or any(answer(d["answer"]) is None for d in docs):
        raise ValueError("Incomplete or unparseable GSM8K test split")
    return docs, dict(repo=GSM8K, revision=GSM8K_REVISION, sha256=common.file_hash(path))


def identity(exp, marker, data):
    from workloads.evaluation import mmlu_generation
    from workloads.instruction import instruct_generation

    paths = (Path(__file__), Path(quality.__file__), Path(instruct_generation.__file__),
             Path(mmlu_generation.__file__))
    return dict(arm=exp.arm, mode=exp.mode, checkpoint=marker["checkpoint_sha256"],
                model_sources=quality.model_sources(exp),
                model=c.MODEL, revision=c.REVISION, generation=source.GENERATION[exp.mode],
                seed=c.SEED, data=data, prompt=INSTRUCTION, code={
                    path.stem: source_hash(path) for path in paths})


def evaluate(exp):
    import torch
    from transformers import AutoTokenizer
    from workloads.instruction.instruct_generation import document_seed, generate

    if torch.cuda.device_count() != 1:
        raise RuntimeError("GSM8K evaluation requires exactly one GPU")
    docs, data = dataset()
    marker = quality.endpoint(exp)
    ident = identity(exp, marker, data)
    root = work_path(ROOT / "runs/out" / exp.name)
    manifest, path = root / "manifest.json", root / "answers.jsonl"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if not compatible_identity(saved, ident, marker, "gsm8k"):
            raise ValueError("Incompatible GSM8K resume")
        ident = saved
    elif path.exists():
        raise ValueError("Orphaned GSM8K answers")
    launch.store_json(manifest, ident)
    rows = recover_samples(path)
    if len(rows) > len(docs):
        raise ValueError("Extra answers")
    for i, row in enumerate(rows):
        if (row["index"] != i or row["question"] != docs[i]["question"]
                or row["gold"] != answer(docs[i]["answer"])
                or row["seed"] != document_seed(c.SEED, "gsm8k", exp.mode, i)
                or row["predicted"] != answer(row["final_text"])
                or row["correct"] != (row["predicted"] == row["gold"])):
            raise ValueError("Mismatched or corrupt saved answers")
    native, packed = quality.load_endpoint(exp, marker)
    del packed
    tokenizer = AutoTokenizer.from_pretrained(c.MODEL, revision=c.REVISION)
    with path.open("a") as stream, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(len(rows), len(docs)):
            doc = docs[i]
            seed = document_seed(c.SEED, "gsm8k", exp.mode, i)
            result = generate(native, tokenizer, [{"role": "user",
                              "content": doc["question"]+"\n\n"+INSTRUCTION}],
                              exp.mode=="thinking", source.GENERATION[exp.mode], seed)
            predicted, gold = answer(result["final_text"]), answer(doc["answer"])
            row = dict(index=i, question=doc["question"], seed=seed, predicted=predicted,
                       gold=gold, correct=predicted==gold, **result)
            stream.write(json.dumps(row)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
            rows.append(row)
    n = len(rows)
    launch.store_json(root / "evaluated.json", dict(identity=ident, count=n,
                      accuracy=sum(r["correct"] for r in rows)/n,
                      invalid_rate=sum(r["predicted"] is None for r in rows)/n,
                      cap_rate=sum(r["stop"]=="length" for r in rows)/n,
                      answers_sha256=common.file_hash(path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter", choices=[e.name for e in experiment_configs])
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    selected = [e for e in experiment_configs if args.filter in (None, e.name)]
    if args.worker:
        exp, = selected
        return evaluate(exp)
    from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile
    _, data = dataset()
    launch.ARTIFACTS_ROOT, launch.RUN_ROOT = ROOT, ROOT / "runs"
    with launch.submission_lock():
        client, opts = training_job_api_from_profile("default")
        launch.disable_submission_retries(client)
        active = {launch.normalize(j["job_desc"]): j["job_name"]
                  for j in get_in_progress_jobs(client_profile="default")}
        launch.reconcile_receipts(client, active)
        launch.reconcile_intents(active)
        commit = launch.git("rev-parse", "HEAD")
        staged = launch.stage_repo(Path(launch.git("rev-parse", "--show-toplevel")), commit)
        for exp in selected:
            marker = quality.endpoint(exp)
            ident = identity(exp, marker, data)
            root = work_path(ROOT / "runs/out" / exp.name)
            if (root / "evaluated.json").exists():
                saved = json.loads((root / "evaluated.json").read_text())
                if (not compatible_identity(saved["identity"], ident, marker, "gsm8k")
                        or saved["answers_sha256"] != common.file_hash(root/"answers.jsonl")):
                    raise ValueError("Stale completion")
                continue
            body = launch.payload("eval", exp.name, staged, "d.tarasov", "mrsndmn")
            body.update(instance_type=launch.INSTANCE_TYPES_BY_NUM_GPUS[1], region=opts["region"],
                        job_desc=f"sps-readout-gsm8k {exp.name} #d.tarasov @mrsndmn",
                        script=f"cd {shlex.quote(str(staged))} && " + shlex.join([
                            "timeout", f"{HOURS[exp.mode]}h", str(launch.PYTHON), "scripts/readout/real_readout_gsm8k.py", "--worker", "--filter", exp.name]))
            body["env_variables"]["SPS_EXPERIMENT_SUITE"] = "ar-sps-qwen3-1.7b-instruct"
            key = launch.normalize(body["job_desc"])
            if key in active:
                print(exp.name, active[key], flush=True)
            else:
                print(json.dumps(launch.submit_payload(client, body, "eval", exp.name, commit)), flush=True)
