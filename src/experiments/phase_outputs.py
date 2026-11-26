"""Configured publication paths for shared adaptation and speculative runs."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PhaseOutputs:
    prepare: str = "prepared"
    train: str = "runs/out/{name}"
    eval: str = "runs/out/{name}/eval"

    by_run: dict[str, dict[str, Path]] = field(default_factory=dict)

    def path(self, root: Path, phase, name=None):
        selected = self.by_run.get(name, {}).get(phase)
        return root / (selected or getattr(self, phase).format(name=name))


adaptation = PhaseOutputs()
speculative = PhaseOutputs(eval="evaluations/{name}")
families = {
    "ar-sps": adaptation,
    "ar-sps-kd": adaptation,
    "ar-sps-360m-ce": adaptation,
    "ar-sps-360m-kd": adaptation,
    "ar-sps-1.7b-ce": adaptation,
    "ar-sps-1.7b-kd": adaptation,
    "ar-sps-qwen3-0.6b": adaptation,
    "ar-sps-qwen3-1.7b": adaptation,
    "ar-sps-qwen3-1.7b-instruct": adaptation,
    "mtp": speculative,
    "mtp-unfreeze": speculative,
    "spec-baselines": speculative,
}
