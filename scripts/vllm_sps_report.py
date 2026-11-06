"""Validate matched vLLM arms and report quality and independent speedup ratios."""

import csv
import json
from statistics import median

from scripts import vllm_sps_common as common
from scripts import vllm_sps_config as c
from scripts.mlspace_launch import store_json


def report():
    results = {}
    manifests = {}
    for exp in c.collect_experiments("eval"):
        root = c.ROOT / "runs/out" / exp.name
        results[exp.name] = json.loads((root / "evaluated.json").read_text())
        manifests[exp.name] = json.loads((root / "manifest.json").read_text())
        if results[exp.name]["design_signature"] != common.signature():
            raise RuntimeError("Cannot combine results from a different design")
    first = next(iter(manifests.values()))
    for manifest in manifests.values():
        for key in (
            "indices",
            "dataset_revision",
            "checkpoint_sha256",
            "gpu",
            "vllm_source",
            "torch_version",
        ):
            if manifest[key] != first[key]:
                raise RuntimeError(f"Unmatched evaluation input: {key}")
    table = []
    for arm in ("ar", "eagle3", "sps", "sps_mtp"):
        row = {"experiment": arm}
        for mode in c.GENERATION:
            name = f"{arm}-{mode}"
            baseline_name = f"{'sps' if arm.startswith('sps') else 'ar'}-{mode}"
            trials = results[name]["trials"]
            baselines = results[baseline_name]["trials"]
            if (
                len(trials) != c.REPEATS
                or len(baselines) != c.REPEATS
                or any(x["count"] != c.EVAL_COUNT for x in trials + baselines)
            ):
                raise RuntimeError("Incomplete benchmark trial")
            speedups = [
                a["decode_tokens_per_second"] / b["decode_tokens_per_second"]
                for a, b in zip(trials, baselines)
            ]
            row.update(
                {
                    f"{mode}_accuracy": trials[0]["accuracy"],
                    f"{mode}_invalid_rate": trials[0]["invalid_rate"],
                    f"{mode}_length_rate": trials[0]["length_rate"],
                    f"{mode}_decode_tokens_per_second": median(
                        x["decode_tokens_per_second"] for x in trials
                    ),
                    f"{mode}_request_seconds": median(
                        x["end_to_end_seconds"] / x["count"] for x in trials
                    ),
                    f"{mode}_matched_verifier_speedup": median(speedups),
                    f"{mode}_speedup_min": min(speedups),
                    f"{mode}_speedup_max": max(speedups),
                }
            )
        table.append(row)
    target = c.ROOT / "reports"
    target.mkdir(parents=True, exist_ok=True)
    store_json(
        target / "results.json",
        {
            "design_signature": common.signature(),
            "rows": table,
            "accuracy_scope": "first timed trial on fixed held-out STEM pilot; other trials remain in per-arm files",
            "speedup": "decode tokens/s divided by matched-verifier baseline tokens/s, paired by trial",
        },
    )
    with (target / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    report()
