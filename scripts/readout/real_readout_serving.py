"""Matched single-request CUDA-graph audit and STEM benchmark for both endpoints."""
import argparse
import gc
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time

from scripts import real_readout_eval as quality
from scripts import real_readout_run as train
from scripts import serving_graphs as graphs
from scripts import serving_runtime as runtime
from scripts import vllm_sps_common as common
from scripts import vllm_sps_config as c
from scripts import mlspace_launch as launch

ROOT = train.ROOT / "serving"
PATCH = Path(__file__).with_name("real_readout_vllm.patch")
experiment_configs = [c.Experiment(f"serving-{arm}", arm) for arm in ("A", "B")]
LAYOUTS = ("sps", "sps_mtp")
AUDIT_PREFIXES = 8


def signature(exp, marker):
    return dict(arm=exp.arm, checkpoint=marker["checkpoint_sha256"],
                graphs=graphs.protocol_signature(),
                evaluator=common.file_hash(Path(__file__)), patch=common.file_hash(PATCH),
                loader=common.file_hash(Path(quality.__file__)),
                analysis={name: common.file_hash(Path(__file__).with_name(name + ".py"))
                          for name in ("draft_sweep", "instruct_generation", "report_mmlu_stem")})


def private_site(folder):
    # Keep every compiled library and all B sources in the pinned graph runtime.
    # Only the A model implementation needs a private writable Python copy.
    site = folder / "site"
    def link(source, destination):
        if str(source).endswith(("qwen3_sps.py", "__init__.py")):
            return shutil.copyfile(source, destination)
        os.symlink(os.path.realpath(source), destination)
    shutil.copytree(graphs.SITE / "vllm", site / "vllm", copy_function=link,
                    ignore=shutil.ignore_patterns("__pycache__"))
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(PATCH.resolve())],
                   cwd=site, check=True)
    return site


def prepare(exp, marker, folder, root):
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Serving preparation requires one GPU")
    _, model = quality.load_endpoint(exp, marker)
    export = folder / "hf"
    if exp.arm == "B":
        common.export(model, export)
    if exp.arm == "A":
        # Export FP32 masters like B. Converting to BF16 belongs to inference.
        model.backbone.save_pretrained(export, safe_serialization=True)
        slots = model.future.detach().cpu().contiguous()
        save_file({"sps_prediction_embeddings": slots}, export / "sps.safetensors")
        index = export / "model.safetensors.index.json"
        if index.exists():
            value = json.loads(index.read_text())
            value["weight_map"]["sps_prediction_embeddings"] = "sps.safetensors"
            value["metadata"]["total_size"] += slots.nbytes
            launch.store_json(index, value)
        config = json.loads((export / "config.json").read_text())
        config.update(architectures=["Qwen3SPSForCausalLM"], native_first_readout=True,
                      sps_horizons=c.HORIZONS)
        launch.store_json(export / "config.json", config)
        AutoTokenizer.from_pretrained(c.MODEL, revision=c.REVISION).save_pretrained(export)
    model.to(torch.bfloat16).eval()
    refs = []
    with torch.inference_mode():
        for row in common.data("val")[:AUDIT_PREFIXES]:
            prefix = row["input_ids"][:c.AUDIT_PREFIX_TOKENS]
            ids = torch.tensor([prefix], device="cuda")
            logits = model(ids, positions=[-1]).logits[0, 0].float()
            values, indices = logits.topk(2, dim=-1)
            refs.append(dict(prefix=prefix, top=indices[:, 0].tolist(),
                             margin=(values[:, 0]-values[:, 1]).tolist()))
    launch.store_json(root / "prefixes.json", refs)
    return export


class ReadoutWorkerExtension(graphs.GraphReplayWorkerExtension):
    def readout_proposals(self, position):
        model = self.model_runner.get_model()
        # One-token generation ends after prefill, leaving its real-prefix rows.
        row = position - int(model.sps_positions[0].item())
        if not 0 <= row < len(model.sps_positions) or int(model.sps_positions[row]) != position:
            raise ValueError("Unexpected cached proposal row")
        return model.compute_logits(model.sps_hidden[row]).argmax(-1).tolist()


def counts(llm):
    return [{"name": m.name, "value": getattr(m, "value", getattr(m, "values", None))}
            for m in llm.get_metrics() if "spec_decode" in m.name]


def engine(exp, layout, export):
    from scripts.vllm_sps_evaluate import check_runtime, engine_arguments
    torch, vllm = check_runtime()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one serving GPU")
    args = graphs.graph_arguments(engine_arguments, c.Experiment(layout, layout), export, None)
    args["worker_extension_cls"] = "scripts.real_readout_serving.ReadoutWorkerExtension"
    return vllm.LLM(**args)


def audit(llm, layout, root):
    from vllm import SamplingParams
    rows = []
    sequential = (json.loads((root / "sps-audit.json").read_text())
                  if layout == "sps_mtp" else [])
    one_token = SamplingParams(temperature=0, max_tokens=1)
    llm.collective_rpc("sps_start_graph_tracking")
    for ref in json.loads((root / "prefixes.json").read_text()):
        prompt = [{"prompt_token_ids": ref["prefix"]}]
        first = llm.generate(prompt, SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
        first_id = first[0].outputs[0].token_ids[0]
        predictions = [first_id]
        if layout == "sps_mtp":
            predictions, = llm.collective_rpc("readout_proposals", args=(len(ref["prefix"])-1,))
            if predictions[0] != first_id:
                raise ValueError("Proposer row disagrees with emitted first token")
        seq = llm.generate(prompt, SamplingParams(temperature=0, max_tokens=c.AUDIT_NEW_TOKENS),
                           use_tqdm=False)[0].outputs[0].token_ids
        evidence = {}
        if layout == "sps":
            evidence["cold_argmax"] = [
                llm.generate([{"prompt_token_ids": ref["prefix"] + list(seq[:i])}],
                             one_token, use_tqdm=False)[0].outputs[0].token_ids[0]
                for i in range(len(seq))]
        else:
            baseline = sequential[len(rows)]
            if list(seq) != baseline["greedy"]:
                i = next((j for j, (a, b) in enumerate(zip(seq, baseline["greedy"])) if a != b),
                         min(len(seq), len(baseline["greedy"])))
                history = ref["prefix"] + list(seq[:i])
                cold = llm.generate([{"prompt_token_ids": history}], one_token,
                                    use_tqdm=False)[0].outputs[0].token_ids[0]
                evidence["first_difference"] = dict(offset=i, history=history, block_cold=cold,
                    sequential_cold=baseline["cold_argmax"][i:i+1],
                    sequential_cached=baseline["greedy"][i:i+1], block_cached=list(seq[i:i+1]))
        rows.append(dict(**evidence, expected=ref["top"], margin=ref["margin"], predicted=predictions,
                         greedy=list(seq)))
    workers = llm.collective_rpc("sps_stop_graph_tracking")
    evidence = root / f"{layout}-graphs.json"
    launch.store_json(evidence, dict(protocol_signature=graphs.protocol_signature(), workers=workers))
    graphs.validate_evidence(evidence)
    launch.store_json(root / f"{layout}-audit.json", rows)
    # Large-margin oracle disagreements flag layout or weight-loading errors.
    for r in rows:
        if len(r["predicted"]) != (c.HORIZONS if layout == "sps_mtp" else 1):
            raise ValueError("Incorrect horizon count")
        if any(actual != oracle and margin > 1. for actual, oracle, margin
               in zip(r["predicted"], r["expected"], r["margin"])):
            raise ValueError("Confident disagreement with the native checkpoint")


def benchmark(llm, exp, layout, root):
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from scripts import instruct_config as source
    from scripts.draft_sweep import counter_metrics
    from scripts.instruct_generation import document_seed, mmlu_messages, parse_answer, split_response
    from scripts.vllm_sps_evaluate import select_examples

    tokenizer = AutoTokenizer.from_pretrained(c.MODEL, revision=c.REVISION)
    docs = list(load_dataset(source.MMLU_DATASET, "all", revision=source.MMLU_REVISION, split="test"))
    indices = select_examples(docs)
    for mode, settings in c.GENERATION.items():
        def generate(index, warm=False):
            params = dict(settings)
            cap = params.pop("max_new_tokens")
            sampling = SamplingParams(**params, max_tokens=min(cap, 32) if warm else cap,
                                      seed=document_seed(c.SEED, "mmlu", mode, index))
            tokens = tokenizer.apply_chat_template(mmlu_messages(docs[index]), tokenize=True,
                        return_dict=False, add_generation_prompt=True, enable_thinking=mode=="thinking")
            torch.cuda.synchronize()
            began = time.perf_counter()
            result = llm.generate([{"prompt_token_ids": tokens}], sampling, use_tqdm=False)[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
            out = result.outputs[0]
            decoded = split_response(tokenizer, list(out.token_ids), mode=="thinking")
            answer = parse_answer(decoded["final_text"])
            return dict(index=index, output_ids=list(out.token_ids), seconds=elapsed,
                        decode_seconds=result.metrics.last_token_ts-result.metrics.first_token_ts,
                        correct=answer == "ABCD"[docs[index]["answer"]], stop=out.finish_reason)

        llm.collective_rpc("sps_start_graph_tracking")
        for index in indices[:c.WARMUP_REQUESTS]:
            generate(index, True)
        evidence = root / f"{layout}-{mode}-graphs.json"
        workers = llm.collective_rpc("sps_stop_graph_tracking")
        launch.store_json(evidence, dict(protocol_signature=graphs.protocol_signature(), workers=workers))
        graphs.validate_evidence(evidence)
        trials = []
        with (root / f"{layout}-{mode}.jsonl").open("w") as stream:
            for repeat in range(c.REPEATS):
                before = counts(llm)
                rows = []
                for index in indices:
                    row = generate(index)
                    rows.append(row)
                    stream.write(json.dumps(dict(repeat=repeat, **row))+"\n")
                seconds = sum(r["decode_seconds"] for r in rows)
                after = counts(llm)
                trials.append(dict(repeat=repeat, decode_seconds=seconds,
                    end_to_end_seconds=sum(r["seconds"] for r in rows),
                    accuracy=sum(r["correct"] for r in rows)/len(rows),
                    decode_tokens_s=sum(max(0, len(r["output_ids"])-1) for r in rows)/seconds,
                    counters_before=before, counters_after=after,
                    acceptance=counter_metrics({"acceptance_counters_including_warmup": after},
                                          c.HORIZONS-1 if layout=="sps_mtp" else 0, before)))
        launch.store_json(root / f"{layout}-{mode}.json", trials)


def worker(exp):
    marker = quality.endpoint(exp)
    root = ROOT / "runs/out" / exp.name
    root.mkdir(parents=True, exist_ok=True)
    graphs.verify_runtime()
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        folder = Path(tmp)
        export = prepare(exp, marker, folder, root)
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        site = private_site(folder) if exp.arm == "A" else graphs.SITE
        env = runtime.runtime_environment()
        env.update(PYTHONPATH=os.pathsep.join([str(site), env["PYTHONPATH"]]),
                   SPS_VLLM_SOURCE=str(site), VLLM_USE_V2_MODEL_RUNNER="0")
        for phase in ("audit", "benchmark"):
            for layout in LAYOUTS:
                subprocess.run([runtime.RUNTIME/"bin/python", "-m", "scripts.real_readout_serving",
                                "--arm", exp.arm, "--internal", phase, "--layout", layout,
                                "--export", export], env=env, check=True)
            if phase == "audit":
                seq, block = [json.loads((root / f"{name}-audit.json").read_text()) for name in LAYOUTS]
                matches = sum(s["greedy"] == b["greedy"] for s, b in zip(seq, block, strict=True))
                audit_passed = matches == len(seq)
                launch.store_json(root / "greedy.json", dict(matches=matches, total=len(seq),
                                  passed=audit_passed))
    launch.store_json(root / "evaluated.json", dict(identity=signature(exp, marker),
                      cached_greedy_passed=audit_passed,
                      timing_scope="paired pilot" if audit_passed else "diagnostic: cached greedy audit failed",
                      artifacts=artifact_hashes(root)))


def artifact_hashes(root):
    names = ["prefixes.json", "greedy.json"]
    for layout in LAYOUTS:
        names.extend((f"{layout}-audit.json", f"{layout}-graphs.json"))
        for mode in c.GENERATION:
            names.extend(f"{layout}-{mode}{suffix}" for suffix in (".json", ".jsonl", "-graphs.json"))
    return {name: common.file_hash(root / name) for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B"))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--internal", choices=("audit", "benchmark"))
    parser.add_argument("--layout", choices=LAYOUTS)
    parser.add_argument("--export")
    args = parser.parse_args()
    selected = [e for e in experiment_configs if args.arm in (None, e.arm)]
    if args.internal or args.worker:
        exp, = selected
        if args.internal:
            llm = engine(exp, args.layout, args.export)
            root = ROOT / "runs/out" / exp.name
            return (audit(llm, args.layout, root) if args.internal == "audit"
                    else benchmark(llm, exp, args.layout, root))
        return worker(exp)
    from mls.manager.job.utils import training_job_api_from_profile, get_in_progress_jobs
    launch.ARTIFACTS_ROOT, launch.RUN_ROOT = ROOT, ROOT / "runs"
    with launch.submission_lock():
        client, options = training_job_api_from_profile("default")
        launch.disable_submission_retries(client)
        active = {launch.normalize(j["job_desc"]): j["job_name"]
                  for j in get_in_progress_jobs(client_profile="default")}
        launch.reconcile_receipts(client, active)
        launch.reconcile_intents(active)
        commit = launch.git("rev-parse", "HEAD")
        staged = launch.stage_repo(Path(launch.git("rev-parse", "--show-toplevel")), commit)
        for exp in selected:
            result = ROOT / "runs/out" / exp.name / "evaluated.json"
            if result.exists():
                saved = json.loads(result.read_text())
                if (saved["identity"] != signature(exp, quality.endpoint(exp))
                        or saved["artifacts"] != artifact_hashes(result.parent)):
                    raise ValueError("Stale serving result")
                continue
            body = launch.payload("eval", exp.name, staged, "d.tarasov", "mrsndmn")
            body.update(instance_type=launch.INSTANCE_TYPES_BY_NUM_GPUS[1], region=options["region"],
                        job_desc=f"sps-real-readout-serving {exp.name} #d.tarasov @mrsndmn",
                        script=f"cd {shlex.quote(str(staged))} && " + shlex.join([
                            "timeout", f"{c.EVAL_HOURS}h", str(launch.PYTHON), "-m",
                            "scripts.real_readout_serving", "--worker", "--arm", exp.arm]))
            key = launch.normalize(body["job_desc"])
            if key in active:
                print(exp.name, active[key], flush=True)
            else:
                print(json.dumps(launch.submit_payload(client, body, "eval", exp.name, commit)), flush=True)


if __name__ == "__main__":
    main()
