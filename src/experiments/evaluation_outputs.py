"""Publication directories for phases with standalone evaluation results."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EvaluationOutputs:
    output_dir: Path

    by_run: dict[str, Path] = field(default_factory=dict)

    def for_run(self, root, run):
        return root / self.by_run.get(
            run.name, Path(str(self.output_dir).format(name=run.name))
        )


families = {
    "mmlu-generation": EvaluationOutputs(Path("mmlu-generation/runs/out/{name}/full")),
    "mmlu-stem-generation": EvaluationOutputs(
        Path("mmlu-stem-generation/runs/out/{name}/full")
    ),
    "mmlu-full-resume": EvaluationOutputs(
        Path("mmlu-full-resume/runs/out/{name}/full")
    ),
    "mtp-parallel": EvaluationOutputs(Path("mtp-parallel/runs/out/{name}")),
    "mtp-unified": EvaluationOutputs(Path("mtp-unified/runs/out/{name}")),
}
