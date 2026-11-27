"""Backbone sizes and training defaults used by the paper recipes."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True, kw_only=True)
class PaperRun:
    name: str
    architecture: str
    tags: tuple[str, ...]
    n_layer: int = 12
    n_head: int = 12
    hidden_size: int = 768
    intermediate_size: int = 2304
    micro_batch_size: int = 6
    block_size: int = 4096
    window_size: int | None = None
    max_tokens: int = 20_000_000_000
    eval_interval_tokens: int = 40_000_000
    save_every: int = 2_000_000_000
    lr_decay_tokens: int = 2_000_000_000
    warmup_tokens: int = 50_000_000
    ddp_timeout_seconds: int | None = None
    seed: int = 1337

    def arguments(self):
        return dict(
            asdict(self), wandb_run_name=self.name, chunk_shuffle_seed=self.seed
        )


@dataclass(frozen=True, kw_only=True)
class LRun(PaperRun):
    n_layer: int = 36
    n_head: int = 20
    hidden_size: int = 1280
    intermediate_size: int = 3840


@dataclass(frozen=True, kw_only=True)
class MRun(PaperRun):
    n_layer: int = 24
    n_head: int = 16
    hidden_size: int = 1024
    intermediate_size: int = 3072


@dataclass(frozen=True, kw_only=True)
class XLRun(PaperRun):
    n_layer: int = 48
    n_head: int = 25
    hidden_size: int = 1600
    intermediate_size: int = 4800


@dataclass(frozen=True, kw_only=True)
class XSRun(PaperRun):
    n_layer: int = 8
    n_head: int = 8
    hidden_size: int = 512
    intermediate_size: int = 1536
