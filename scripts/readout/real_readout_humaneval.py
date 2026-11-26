"""Generate one HumanEval+ solution per task for external EvalPlus scoring."""
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

from scripts import mlspace_launch as launch
from scripts import real_readout_eval as quality
from scripts import real_readout_run as train
from scripts import vllm_sps_common as common
from scripts import vllm_sps_config as c
from scripts.mmlu_results import recover_samples

ROOT = train.ROOT / "humaneval-plus"
PROMPTS = ROOT / "input/prompts.jsonl"
RELEASE = "v0.1.10"
URL = f"https://github.com/evalplus/humanevalplus_release/releases/download/{RELEASE}/HumanEvalPlus.jsonl.gz"
ARCHIVE_SHA = "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101"
PROMPT_SHA = "2daaf2edc5a3dc6a68316a89800b61af7d534bca5f6230a5e6b6773411576c0a"
REQUEST = ("Implement the Python function described below. Include its signature, all required "
           "imports and any helper functions. Return the complete solution in one Python code block.")
SETTINGS = {mode: dict(settings, max_new_tokens=4096) for mode, settings in c.GENERATION.items()}
experiment_configs = [c.Experiment(f"{arm}-{mode}", arm, mode) for arm in ("A", "B") for mode in SETTINGS]
HOURS = 16
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
                settings=SETTINGS[exp.mode], seed=c.SEED, request=REQUEST,
                code={s: common.file_hash(Path(__file__).with_name(s+".py")) for s in
                      ("real_readout_humaneval", "real_readout_eval", "instruct_generation", "mmlu_generation")})


def paths(exp):
    return ROOT/"runs/out"/exp.name, ROOT/"samples"/(exp.name+".jsonl")


def evaluate(exp):
    import torch
    from transformers import AutoTokenizer
    from scripts.instruct_generation import document_seed, generate

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one GPU")
    docs = dataset()
    marker = quality.endpoint(exp)
    ident = identity(exp, marker)
    root, samples = paths(exp)
    meta, raw = root/"manifest.json", root/"raw.jsonl"
    if meta.exists():
        if json.loads(meta.read_text()) != ident:
            raise ValueError("HumanEval+ manifest mismatch")
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
    from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile
    launch.ARTIFACTS_ROOT, launch.RUN_ROOT = ROOT, ROOT/"runs"
    with launch.submission_lock():
        dataset(prepare=True)
        client, options = training_job_api_from_profile("default")
        launch.disable_submission_retries(client)
        active = {launch.normalize(j["job_desc"]): j["job_name"]
                  for j in get_in_progress_jobs(client_profile="default")}
        launch.reconcile_receipts(client, active)
        launch.reconcile_intents(active)
        commit = launch.git("rev-parse", "HEAD")
        staged = launch.stage_repo(Path(launch.git("rev-parse", "--show-toplevel")), commit)
        for exp in selected:
            root, samples = paths(exp)
            ident = identity(exp, quality.endpoint(exp))
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
                            "timeout", f"{HOURS}h", str(launch.PYTHON), "-m",
                            "scripts.real_readout_humaneval", "--worker", "--filter", exp.name]))
            body["env_variables"]["SPS_EXPERIMENT_SUITE"] = "ar-sps-qwen3-1.7b-instruct"
            key = launch.normalize(body["job_desc"])
            if key in active:
                print(exp.name, active[key], flush=True)
            else:
                print(json.dumps(launch.submit_payload(client, body, "eval", exp.name, commit)), flush=True)


if __name__ == "__main__":
    main()
