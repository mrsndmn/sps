"""Generate one HumanEval+ solution per task for external EvalPlus scoring."""
from orchestration.named.publication import work_path
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import tempfile
from urllib.request import urlopen

from orchestration import mlspace_launch as launch
from workloads.readout import real_readout_eval as quality
from workloads.readout import real_readout_run as train
from workloads.serving import vllm_sps_common as common
from experiments import vllm_sps_config as c
from workloads.evaluation.mmlu_results import recover_samples
from experiments.real_readout_humaneval import (
    ROOT,
    PROMPTS,
    RELEASE,
    URL,
    ARCHIVE_SHA,
    PROMPT_SHA,
    REQUEST,
    SETTINGS,
    experiment_configs,
    HOURS,
)


FENCE = re.compile(r"^[ \t]*```(?:python|py)?[ \t]*\r?\n(.*?)(?:^[ \t]*```[ \t]*\r?$|\Z)",
                   re.IGNORECASE | re.MULTILINE | re.DOTALL)


def dataset(prepare=False):
    if prepare and not PROMPTS.exists():
        with urlopen(URL, timeout=60) as response:
            archive = response.read()
        if hashlib.sha256(archive).hexdigest() != ARCHIVE_SHA:
            raise ValueError("HumanEval+ release checksum mismatch")
        docs = [json.loads(line) for line in gzip.decompress(archive).splitlines()]
        text = "".join(json.dumps({k: doc[k] for k in ("task_id", "prompt", "entry_point")},
                                  sort_keys=True)+"\n" for doc in docs)
        if hashlib.sha256(text.encode()).hexdigest() != PROMPT_SHA:
            raise ValueError("Unexpected prompt projection")
        PROMPTS.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=PROMPTS.parent, delete=False) as tmp:
            tmp.write(text.encode())
        Path(tmp.name).replace(PROMPTS)
    if common.file_hash(PROMPTS) != PROMPT_SHA:
        raise ValueError("Unverified HumanEval+ prompt input")
    docs = [json.loads(s) for s in PROMPTS.read_text().splitlines()]
    if len(docs) != 164 or {d["task_id"] for d in docs} != {f"HumanEval/{i}" for i in range(164)}:
        raise ValueError("Incomplete HumanEval+ tasks")
    return docs


def solution(final_text):
    match = FENCE.search(final_text)
    return match.group(1).rstrip()+"\n" if match else final_text


def identity(exp, marker):
    return dict(arm=exp.arm, mode=exp.mode, checkpoint=marker["checkpoint_sha256"],
                prompts=PROMPT_SHA, release=RELEASE,
                settings=SETTINGS[exp.mode], seed=c.SEED, request=REQUEST)


def paths(exp):
    output = work_path(ROOT / "runs/out" / exp.name)
    legacy = ROOT / "samples" / (exp.name + ".jsonl")
    samples = output / "samples.jsonl"
    return output, legacy if legacy.is_file() and output == ROOT/"runs/out"/exp.name and not samples.exists() else samples


def evaluate(exp):
    import torch
    from transformers import AutoTokenizer
    from workloads.instruction.instruct_generation import document_seed, generate

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one GPU")
    docs = dataset()
    marker = quality.endpoint(exp)
    ident = identity(exp, marker)
    root, samples = paths(exp)
    meta, raw = root/"manifest.json", root/"raw.jsonl"
    if meta.exists():
        saved = json.loads(meta.read_text())
        if saved != ident:
            raise ValueError("HumanEval+ manifest mismatch")
        ident = saved
    elif raw.exists() or samples.exists():
        raise ValueError("Missing HumanEval+ manifest")
    launch.store_json(meta, ident)
    rows = recover_samples(raw)
    def seed(doc):
        return document_seed(c.SEED, "humaneval_plus", exp.mode, int(doc["task_id"].split("/")[1]))
    if len(rows) > len(docs) or any(row["task_id"] != doc["task_id"] or row["seed"] != seed(doc)
                                    for row, doc in zip(rows, docs)):
        raise ValueError("Invalid saved generations")
    if len(rows) < len(docs):
        model, packed = quality.load_endpoint(exp, marker)
        del packed
        tokenizer = AutoTokenizer.from_pretrained(c.MODEL, revision=c.REVISION)
        with raw.open("a") as stream, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for doc in docs[len(rows):]:
                content = REQUEST+"\n\n```python\n"+doc["prompt"].strip()+"\n```"
                response = generate(model, tokenizer, [{"role": "user", "content": content}],
                                    exp.mode == "thinking", SETTINGS[exp.mode], seed(doc))
                row = dict(task_id=doc["task_id"], seed=seed(doc), **response)
                stream.write(json.dumps(row)+"\n")
                stream.flush()
                os.fsync(stream.fileno())
                rows.append(row)
    samples.parent.mkdir(parents=True, exist_ok=True)
    tmp = samples.with_suffix(".tmp")
    with tmp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(dict(task_id=row["task_id"], solution=solution(row["final_text"])))+"\n")
    tmp.replace(samples)
    launch.store_json(root/"evaluated.json", dict(identity=ident, generated=len(rows),
                      cap_count=sum(r["stop"] == "length" for r in rows),
                      empty_count=sum(not solution(r["final_text"]).strip() for r in rows),
                      samples_sha256=common.file_hash(samples), raw_sha256=common.file_hash(raw)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter", choices=[e.name for e in experiment_configs])
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    selected = [e for e in experiment_configs if args.filter in (None, e.name)]
    if args.worker:
        exp, = selected
        return evaluate(exp)
    selected = [exp for exp in selected if launch.pending(ROOT / "runs/out" / exp.name)]
    if not selected:
        return
    from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile
    launch.ARTIFACTS_ROOT, launch.RUN_ROOT = ROOT, ROOT/"runs"
    with launch.submission_lock():
        dataset(prepare=True)
        client, options = training_job_api_from_profile("default")
        launch.disable_submission_retries(client)
        active = {launch.normalize(j["job_desc"]): j["job_name"]
                  for j in get_in_progress_jobs(client_profile="default")}
        launch.reconcile_for_outputs(client, active, [ROOT / "runs/out" / exp.name for exp in selected])
        commit = launch.git("rev-parse", "HEAD")
        staged = launch.stage_repo(Path(launch.git("rev-parse", "--show-toplevel")), commit)
        for exp in selected:
            root, samples = paths(exp)
            marker = quality.endpoint(exp)
            ident = identity(exp, marker)
            if (root/"evaluated.json").exists():
                done = json.loads((root/"evaluated.json").read_text())
                if (done["identity"] != ident or done["generated"] != 164
                        or done["samples_sha256"] != common.file_hash(samples)
                        or done["raw_sha256"] != common.file_hash(root/"raw.jsonl")):
                    raise ValueError("Stale HumanEval+ result")
                continue
            body = launch.payload("eval", exp.name, staged, "d.tarasov", "mrsndmn")
            body.update(instance_type=launch.INSTANCE_TYPES_BY_NUM_GPUS[1], region=options["region"],
                        job_desc=f"sps-readout-humaneval-plus {exp.name} #d.tarasov @mrsndmn",
                        script=f"cd {shlex.quote(str(staged))} && " + shlex.join([
                            "timeout", f"{HOURS}h", str(launch.PYTHON), "scripts/readout/real_readout_humaneval.py", "--worker", "--filter", exp.name]))
            body["env_variables"]["SPS_EXPERIMENT_SUITE"] = "ar-sps-qwen3-1.7b-instruct"
            key = launch.output_identity(body["job_desc"], ROOT / "runs/out" / exp.name, active)
            if key in active:
                print(exp.name, active[key], flush=True)
            else:
                print(json.dumps(launch.submit_payload(client, body, "eval", exp.name, commit, output_dir=ROOT / "runs/out" / exp.name)), flush=True)
