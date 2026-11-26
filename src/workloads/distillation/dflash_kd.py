"""Submit and report the DFlash soft-distillation experiment."""

import json
from pathlib import Path
import tempfile

from workloads.distillation import dflash_kd_train as training, dflash_repro as runner
from workloads.distillation import dflash_repro_eval as evaluation
from workloads.distillation import dflash_repro_train as hard
from orchestration import mlspace_launch as launch


def evaluate(checkpoint, result, *, experiment):
    # Reuse the exact hard-run evaluation protocol without changing its archived identity.
    result.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=result.parent) as tmp:
        raw = Path(tmp) / "raw.json"
        evaluation.evaluate(checkpoint, raw, experiment=experiment)
        report = json.loads(raw.read_text())
        report.update(
            design=experiment.name,
            evaluation=evaluation.signature(),
            hard_design=hard.EXPERIMENT.name,
            objective="soft-kl",
        )
        launch.store_json(result, report)


def main():
    runner.run(training, "scripts/distillation/dflash_kd.py", evaluate)
