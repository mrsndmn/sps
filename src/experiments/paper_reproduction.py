"""FineWeb preparation and paper-scale zero-window SPS training runs."""
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from experiments.environment import COMMON_ROOT
from experiments.launch import Launch
from experiments.paper_models import MRun, PaperRun, XLRun
from experiments.pilot import data_files


@dataclass(frozen=True, kw_only=True)
class PaperSPS:
    name: str
    num_gpus: int
    family: ClassVar[str] = "paper-sps"
    script: ClassVar[str] = "scripts/pilot/paper_reproduction.py"
    phase: ClassVar[str]

    @classmethod
    def experiment_configs(cls) -> list["PrepareFineWeb | TrainSPS"]:
        corpus = PrepareFineWeb(name="data", num_gpus=1)
        return [
            corpus,
            TrainSPS(name="s", corpus=corpus, model=PaperRun, warmup_tokens=50_000_000, micro_batch_size=6),
            TrainSPS(name="m", corpus=corpus, model=MRun, warmup_tokens=100_000_000, micro_batch_size=2),
            TrainSPS(name="xl", corpus=corpus, model=XLRun, warmup_tokens=100_000_000, micro_batch_size=1),
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


@dataclass(frozen=True, kw_only=True)
class TrainSPS(PaperSPS):
    phase = "train"
    corpus: PrepareFineWeb
    model: type[PaperRun]
    warmup_tokens: int
    micro_batch_size: int
    num_gpus: int = 4

    @property
    def recipe(self) -> PaperRun:
        window_tags = () if self.model is XLRun else ("sliding_predict_tokens",)
        return self.model(
            name=f"repro_{self.name}_sps_w0_20b",
            architecture="sps",
            tags=(self.name, "sps", *window_tags, "deterministic", "w0", "20b", "triton"),
            window_size=0,
            warmup_tokens=self.warmup_tokens,
            micro_batch_size=self.micro_batch_size,
        )

    @property
    def output(self):
        return self.corpus.data_root / "out" / self.recipe.name
