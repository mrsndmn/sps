"""Full, zero-shot MMLU chat generation for the final native and SPS readouts."""
from experiments.family_cli import worker_child
from orchestration.named.publication import work_path
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from experiments import instruct_config as source
from orchestration import mlspace_launch as launch
from workloads.readout import real_readout_eval as quality
from workloads.readout import real_readout_serving as serving
from workloads.serving import vllm_sps_common as common
from experiments import vllm_sps_config as c
from experiments.real_readout_mmlu import family_config as real_readout_mmlu_settings
from experiments.real_readout_mmlu import experiment_configs
ROOT = real_readout_mmlu_settings.ROOT
HOURS = real_readout_mmlu_settings.HOURS
COUNT = real_readout_mmlu_settings.COUNT
SUBJECTS = real_readout_mmlu_settings.SUBJECTS


def dataset():
    from experiments.environment import HF_HOME
    from huggingface_hub import hf_hub_download
    from pyarrow.parquet import read_table
    path = hf_hub_download(source.family_config.MMLU_DATASET, "all/test-00000-of-00001.parquet",
                           revision=source.family_config.MMLU_REVISION, cache_dir=HF_HOME/"hub",
                           repo_type="dataset", local_files_only=True)
    docs = read_table(path).to_pylist()
    if len(docs) != COUNT or len({d["subject"] for d in docs}) != SUBJECTS:
        raise ValueError("Incomplete full MMLU test split")
    return docs, common.file_hash(path)


def identity(exp, marker, digest):
    return dict(serving=serving.signature(exp, marker), mode=exp.mode, layout="sequential",
                model=c.family_config.MODEL, revision=c.family_config.REVISION, seed=c.family_config.SEED, settings=c.family_config.GENERATION[exp.mode],
                dataset=source.family_config.MMLU_DATASET, dataset_revision=source.family_config.MMLU_REVISION,
                split="all/test", dataset_sha256=digest)


def score(row, index, doc, mode, stats):
    from workloads.instruction.instruct_generation import document_seed, parse_answer
    target = "ABCD"[doc["answer"]]
    prediction = parse_answer(row["final_text"])
    if (row["index"] != index or row["subject"] != doc["subject"] or row["target"] != target
            or row["answer"] != prediction or row["seed"] != document_seed(c.family_config.SEED, "mmlu", mode, index)
            or row["correct"] != (prediction == target)):
        raise ValueError("Saved MMLU answer does not match its question or score")
    stats.update(count=1, correct=int(row["correct"]), invalid=int(prediction is None),
                 capped=int(row["stop"] == "length"))
    stats["count/"+doc["subject"]] += 1
    stats["correct/"+doc["subject"]] += int(row["correct"])


def recover(path, docs, mode, stats):
    if not path.exists():
        return
    with path.open("rb+") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                stream.truncate(offset)
                break
            index = stats["count"]
            if index >= len(docs):
                raise ValueError("Extra MMLU answer")
            score(json.loads(line), index, docs[index], mode, stats)


def generate(exp, export):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from workloads.instruction.instruct_generation import document_seed, mmlu_messages, parse_answer, split_response

    docs, digest = dataset()
    root = work_path(ROOT / "runs/out" / exp.name)
    marker = quality.endpoint(exp)
    ident = json.loads((root/"manifest.json").read_text())
    if ident != identity(exp, marker, digest):
        raise ValueError("Export or evaluation identity changed")
    model = serving.engine(exp, "sps", export)
    tokenizer = AutoTokenizer.from_pretrained(c.family_config.MODEL, revision=c.family_config.REVISION)
    stats = Counter()
    path = root/"answers.jsonl"
    recover(path, docs, exp.mode, stats)
    settings = dict(c.family_config.GENERATION[exp.mode])
    cap = settings.pop("max_new_tokens")
    thinking = exp.mode == "thinking"
    with path.open("a") as stream:
        for i in range(stats["count"], len(docs)):
            doc = docs[i]
            seed = document_seed(c.family_config.SEED, "mmlu", exp.mode, i)
            ids = tokenizer.apply_chat_template(mmlu_messages(doc), tokenize=True, return_dict=False,
                                                 add_generation_prompt=True, enable_thinking=thinking)
            if len(ids) + cap > c.family_config.MAX_MODEL_LEN:
                raise ValueError(f"Question {i} exceeds serving context")
            params = SamplingParams(**settings, max_tokens=cap, seed=seed)
            output = model.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0].outputs[0]
            tokens = list(output.token_ids)
            response = split_response(tokenizer, tokens, thinking)
            prediction = parse_answer(response["final_text"])
            target = "ABCD"[doc["answer"]]
            row = dict(index=i, seed=seed, subject=doc["subject"], answer=prediction, target=target,
                       correct=prediction==target, stop=output.finish_reason, output_ids=tokens, **response)
            score(row, i, doc, exp.mode, stats)
            stream.write(json.dumps(row)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
    if stats["count"] != COUNT:
        raise ValueError("Incomplete test set")
    subjects = {s: dict(count=stats["count/"+s], correct=stats["correct/"+s])
                for s in sorted({d["subject"] for d in docs})}
    launch.store_json(root/"evaluated.json", dict(identity=ident, count=COUNT,
                      accuracy=stats["correct"]/COUNT, by_subject=subjects,
                      macro_accuracy=sum(v["correct"]/v["count"] for v in subjects.values())/SUBJECTS,
                      invalid_rate=stats["invalid"]/COUNT, cap_rate=stats["capped"]/COUNT,
                      answers_sha256=common.file_hash(path)))


def worker(exp):
    import torch
    root = work_path(ROOT / "runs/out" / exp.name)
    marker = quality.endpoint(exp)
    _, digest = dataset()
    ident = identity(exp, marker, digest)
    meta = root/"manifest.json"
    if meta.exists():
        saved = json.loads(meta.read_text())
        if saved != ident:
            raise ValueError("Incompatible MMLU resume")
        ident = saved
    elif (root/"answers.jsonl").exists():
        raise ValueError("Missing answer manifest")
    launch.store_json(meta, ident)
    serving.graphs.verify_runtime()
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        folder = Path(tmp)
        export = serving.prepare(exp, marker, folder, root)
        gc.collect()
        torch.cuda.empty_cache()
        site = serving.private_site(folder) if exp.arm == "A" else serving.graphs.SITE
        env = serving.runtime.runtime_environment()
        env.update(PYTHONPATH=os.pathsep.join([str(site), env["PYTHONPATH"]]),
                   SPS_VLLM_SOURCE=str(site), VLLM_USE_V2_MODEL_RUNNER="0")
        subprocess.run(worker_child([serving.runtime.RUNTIME/"bin/python", "scripts/readout/real_readout_mmlu.py",
                        "--filter", exp.name, "--export", export]), env=env, check=True)


def payload(exp, staged):
    body = launch.payload("eval", exp.name, staged, "d.tarasov", "mrsndmn")
    body.update(instance_type=launch.INSTANCE_TYPES_BY_NUM_GPUS[1],
                job_desc=f"sps-readout-full-mmlu {exp.name} #d.tarasov @mrsndmn",
                script=f"cd {shlex.quote(str(staged))} && " + shlex.join([
                    "timeout", f"{HOURS}h", str(launch.PYTHON), "scripts/readout/real_readout_mmlu.py", "--worker", "--filter", exp.name]))
    body["env_variables"]["SPS_EXPERIMENT_SUITE"] = "ar-sps-qwen3-1.7b-instruct"
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter", choices=[e.name for e in experiment_configs])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--export")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    selected = [e for e in experiment_configs if args.filter in (None, e.name)]
    if args.dry:
        staged = launch.STAGING_ROOT / launch.git("rev-parse", "HEAD")
        for exp in selected:
            launch.print_preview(payload(exp, staged), ROOT / "runs/out" / exp.name, staged, "eval")
        return
    if args.worker or args.export:
        exp, = selected
        return generate(exp, args.export) if args.export else worker(exp)
    selected = [exp for exp in selected if launch.pending(ROOT / "runs/out" / exp.name)]
    if not selected:
        return
    _, digest = dataset()
    launch.ARTIFACTS_ROOT, launch.RUN_ROOT = ROOT, ROOT/"runs"
    with launch.submission_lock():
        commit = launch.git("rev-parse", "HEAD")
        batch = launch.SubmissionBatch(commit)
        staged = launch.STAGING_ROOT / commit
        for exp in selected:
            root = work_path(ROOT/"runs/out"/exp.name)
            marker = quality.endpoint(exp)
            ident = identity(exp, marker, digest)
            if (root/"evaluated.json").exists():
                done = json.loads((root/"evaluated.json").read_text())
                if (done["identity"] != ident or done["count"] != COUNT
                        or done["answers_sha256"] != common.file_hash(root/"answers.jsonl")):
                    raise ValueError("Stale full MMLU result")
                continue
            body = payload(exp, staged)
            print(json.dumps(batch.submit(body, "eval", exp.name, ROOT / "runs/out" / exp.name)), flush=True)
