"""FineWeb preparation and paper-scale zero-window SPS training runs."""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from hashlib import sha256
from typing import ClassVar

from experiments.environment import COMMON_ROOT
from experiments.launch import Launch
from experiments.paper_models import MRun, PaperRun, XLRun
from experiments.pilot import DATASET_REVISION, data_files
from experiments.worker_args import TrainArgs


@dataclass(frozen=True, kw_only=True)
class PaperSPS:
    name: str
    num_gpus: int
    family: ClassVar[str] = "paper-sps"
    script: ClassVar[str] = "scripts/pilot/paper_reproduction.py"
    phase: ClassVar[str]

    @classmethod
    def experiment_configs(cls) -> list["PrepareFineWeb | TokenizePart | TrainSPS"]:
        corpus = PrepareFineWeb(name="data", num_gpus=1)
        xl = TrainSPS(name="xl", corpus=corpus, model=XLRun, warmup_tokens=100_000_000,
                      micro_batch_size=1, ddp_timeout_seconds=3600)
        probe = XLProbe(name="xl-4x4-probe", corpus=corpus, model=xl.model,
                        warmup_tokens=xl.warmup_tokens, micro_batch_size=xl.micro_batch_size,
                        num_nodes=4, num_gpus=16, chunk_size_units=1024,
                        validation_batch_sequences=24, ddp_timeout_seconds=300)
        return [
            corpus,
            *corpus.parts,
            TrainSPS(name="s", corpus=corpus, model=PaperRun, warmup_tokens=50_000_000, micro_batch_size=6),
            TrainSPS(name="m", corpus=corpus, model=MRun, warmup_tokens=100_000_000, micro_batch_size=2),
            xl, probe,
            replace(xl, name="xl-4x4", num_nodes=probe.num_nodes, num_gpus=probe.num_gpus,
                    chunk_size_units=probe.chunk_size_units,
                    validation_batch_sequences=probe.validation_batch_sequences, probe=probe),
        ]

    @classmethod
    def launches(cls, family):
        for run in cls.experiment_configs():
            yield Launch(run.name, run.phase, cls.script, (run.name,))


@dataclass(frozen=True, kw_only=True)
class PrepareFineWeb(PaperSPS):
    phase = "prepare"
    data_root: Path = COMMON_ROOT / "paper-sps-scale-reproduction"
    prefix: Path = COMMON_ROOT / "10b/runs/data/fineweb-edu"
    prefix_files: tuple[str, ...] = data_files("10b")
    extra_files: tuple[str, ...] = (
        *(f"sample/100BT/{i:03d}_00001.parquet" for i in range(2, 14)),
        *(f"sample/100BT/{i:03d}_00002.parquet" for i in range(8)),
    )
    free_bytes: int = 120_000_000_000

    @property
    def files(self):
        return self.prefix_files + self.extra_files

    @property
    def output(self):
        return self.data_root / "data" / "fineweb-edu"

    @property
    def parts(self):
        groups = (self.extra_files[:8], self.extra_files[8:12],
                  self.extra_files[12:16], self.extra_files[16:])
        return [TokenizePart(name=f"part-{i}", corpus=self, files=files, num_gpus=1)
                for i, files in enumerate(groups, 1)]


@dataclass(frozen=True, kw_only=True)
class TokenizePart(PaperSPS):
    phase = "prepare"
    corpus: PrepareFineWeb
    files: tuple[str, ...]

    @property
    def output(self):
        key = sha256("\n".join(self.files).encode()).hexdigest()[:16]
        return self.corpus.data_root / "parts" / f"{self.name}-{key}"


@dataclass(frozen=True, kw_only=True)
class TrainSPS(PaperSPS):
    phase = "train"
    corpus: PrepareFineWeb
    model: type[PaperRun]
    warmup_tokens: int
    micro_batch_size: int
    num_gpus: int = 4
    ddp_timeout_seconds: int | None = None

    num_nodes: int = 1
    chunk_size_units: int | None = None
    validation_batch_sequences: int | None = None
    stop_after_steps: int | None = None
    probe: "XLProbe | None" = None

    @property
    def recipe(self) -> PaperRun:
        window_tags = () if self.model is XLRun else ("sliding_predict_tokens",)
        return self.model(
            name=f"repro_{self.name}_sps_w0_20b",
            architecture="sps",
            tags=(self.name, "sps", *window_tags, "deterministic", "w0", "20b", "triton"),
            window_size=0,
            distributed_validation=True,
            warmup_tokens=self.warmup_tokens,
            micro_batch_size=self.micro_batch_size,
            ddp_timeout_seconds=self.ddp_timeout_seconds,
            chunk_size_units=self.chunk_size_units,
            validation_batch_sequences=self.validation_batch_sequences,
            stop_after_steps=self.stop_after_steps,
        )

    def benchmark_workload(self):
        recipe = self.recipe.arguments()
        for field in ("name", "tags", "wandb_run_name", "stop_after_steps", "ddp_timeout_seconds"):
            recipe.pop(field)
        settings = asdict(TrainArgs())
        settings.update({key: value for key, value in recipe.items() if value is not None})
        return dict(settings=settings, num_gpus=self.num_gpus, num_nodes=self.num_nodes,
                    corpus=str(self.corpus.output), files=self.corpus.files, revision=DATASET_REVISION)

    @property
    def output(self):
        return self.corpus.data_root / "out" / self.recipe.name


@dataclass(frozen=True, kw_only=True)
class XLProbe(TrainSPS):
    phase = "probe"
    stop_after_steps: int = 32
    discard_steps: int = 5
    baseline_step_seconds: float = 23.2
    baseline_validation_seconds: float = 170.0

    @property
    def fingerprint(self):
        config = dict(workload=self.benchmark_workload(), steps=self.stop_after_steps,
                      discard_steps=self.discard_steps, timeout=self.ddp_timeout_seconds,
                      baseline_step_seconds=self.baseline_step_seconds,
                      baseline_validation_seconds=self.baseline_validation_seconds)
        return sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

    @property
    def output(self):
        path = super().output
        return path.with_name(f"{path.name}-{self.fingerprint[:16]}")
