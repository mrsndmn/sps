"""Submit and report the DFlash soft-distillation experiment."""

import json
from pathlib import Path
from dataclasses import replace
import tempfile

from workloads.distillation import dflash_kd_train as training, dflash_repro as runner
from workloads.distillation import dflash_repro_eval as evaluation
from experiments.dflash_repro_train import HARD_BASELINE_NAME
from orchestration import mlspace_launch as launch


def evaluate(args):
    result = args.result
    # Reuse the exact hard-run evaluation protocol without changing its archived identity.
    result.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=result.parent) as tmp:
        raw = Path(tmp) / "raw.json"
        evaluation.evaluate(replace(args, result=raw))
        report = json.loads(raw.read_text())
        report.update(
            design=args.name,
            evaluation=evaluation.signature(),
            hard_design=HARD_BASELINE_NAME,
            objective="soft-kl",
        )
        launch.store_json(result, report)


def main():
    runner.run(training, evaluate)
