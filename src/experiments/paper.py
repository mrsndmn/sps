"""Named recipes for the paper training, evaluation and generation commands."""

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
    window_size: int | None = None
    max_tokens: int = 20_000_000_000
    eval_interval_tokens: int = 40_000_000
    save_every: int = 2_000_000_000
    lr_decay_tokens: int = 2_000_000_000
    warmup_tokens: int = 50_000_000
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


experiment_configs = [
    LRun(
        name="l_delayed_state_w64_20b",
        architecture="delayed_state",
        tags=(
            "l",
            "delayed_state",
            "sliding_predict_tokens",
            "deterministic",
            "w64",
            "20b",
            "triton",
        ),
        micro_batch_size=3,
        warmup_tokens=100000000,
        window_size=64,
    ),
    LRun(
        name="l_full_attention_20b",
        architecture="standard",
        tags=("l", "full_attention", "20b"),
        warmup_tokens=100000000,
    ),
    LRun(
        name="l_full_attention_40b",
        architecture="standard",
        tags=("l", "full_attention", "40b"),
        max_tokens=40000000000,
        lr_decay_tokens=4000000000,
        warmup_tokens=100000000,
    ),
    LRun(
        name="l_sps_w4096_20b",
        architecture="sps",
        tags=(
            "l",
            "sps",
            "sliding_predict_tokens",
            "deterministic",
            "w4096",
            "20b",
            "triton",
        ),
        micro_batch_size=3,
        warmup_tokens=100000000,
        window_size=4096,
    ),
    LRun(
        name="l_sps_w64_20b",
        architecture="sps",
        tags=(
            "l",
            "sps",
            "sliding_predict_tokens",
            "deterministic",
            "w64",
            "20b",
            "triton",
        ),
        micro_batch_size=3,
        warmup_tokens=100000000,
        window_size=64,
    ),
    MRun(
        name="m_delayed_state_w64_20b",
        architecture="delayed_state",
        tags=(
            "m",
            "delayed_state",
            "sliding_predict_tokens",
            "deterministic",
            "w64",
            "20b",
            "triton",
        ),
        warmup_tokens=100000000,
        window_size=64,
    ),
    MRun(
        name="m_full_attention_20b",
        architecture="standard",
        tags=("m", "full_attention", "20b"),
        warmup_tokens=100000000,
    ),
    MRun(
        name="m_full_attention_40b",
        architecture="standard",
        tags=("m", "full_attention", "40b"),
        max_tokens=40000000000,
        lr_decay_tokens=4000000000,
        warmup_tokens=100000000,
    ),
    MRun(
        name="m_sps_w4096_20b",
        architecture="sps",
        tags=(
            "m",
            "sps",
            "sliding_predict_tokens",
            "deterministic",
            "w4096",
            "20b",
            "triton",
        ),
        warmup_tokens=100000000,
        window_size=4096,
    ),
    MRun(
        name="m_sps_w64_20b",
        architecture="sps",
        tags=(
            "m",
            "sps",
            "sliding_predict_tokens",
            "deterministic",
            "w64",
            "20b",
            "triton",
        ),
        warmup_tokens=100000000,
        window_size=64,
    ),
    PaperRun(
        name="s_delayed_state_w0_20b",
        architecture="delayed_state",
        tags=(
            "s",
            "delayed_state",
            "sliding_predict_tokens",
            "deterministic",
            "w0",
            "20b",
            "triton",
        ),
        window_size=0,
    ),
    PaperRun(
        name="s_delayed_state_w16_20b",
        architecture="delayed_state",
        tags=(
            "s",
            "delayed_state",
            "sliding_predict_tokens",
            "deterministic",
            "w16",
            "20b",
            "triton",
        ),
        window_size=16,
    ),
    PaperRun(
        name="s_delayed_state_w256_20b",
        architecture="delayed_state",
        tags=(
            "s",
            "delayed_state",
            "sliding_predict_tokens",
            "deterministic",
            "w256",
            "20b",
            "triton",
        ),
        window_size=256,
    ),
    PaperRun(
        name="s_delayed_state_w64_10b",
        architecture="delayed_state",
        tags=(
            "s",
            "delayed_state",
            "sliding_predict_tokens",
            "deterministic",
            "w64",
            "10b",
            "triton",
        ),
        max_tokens=10000000000,
        eval_interval_tokens=20000000,
        save_every=1000000000,
        lr_decay_tokens=1000000000,
        window_size=64,
    ),
    PaperRun(
        name="s_delayed_state_w64_10b_seed0",
        architecture="delayed_state",
        tags=(
            "s",
            "delayed_state",
            "sliding_predict_tokens",
            "deterministic",
            "w64",
            "10b",
            "triton",
            "seed_sweep",
            "seed0",
        ),
        max_tokens=10000000000,
        eval_interval_tokens=20000000,
        save_every=1000000000,
        seed=0,
        lr_decay_tokens=1000000000,
        window_size=64,
    ),
    PaperRun(
        name="s_delayed_state_w64_10b_seed1",
        architecture="delayed_state",
        tags=(
            "s",
            "delayed_state",
            "sliding_predict_tokens",
            "deterministic",
            "w64",
            "10b",
            "triton",
            "seed_sweep",
            "seed1",
        ),
        max_tokens=10000000000,
        eval_interval_tokens=20000000,
        save_every=1000000000,
        seed=1,
        lr_decay_tokens=1000000000,
        window_size=64,
    ),
    PaperRun(
        name="s_delayed_state_w64_20b",
        architecture="delayed_state",
        tags=(
            "s",
            "delayed_state",
            "sliding_predict_tokens",
            "deterministic",
            "w64",
            "20b",
            "triton",
        ),
        window_size=64,
    ),
    PaperRun(
        name="s_full_attention_10b",
        architecture="standard",
        tags=("s", "full_attention", "10b"),
        max_tokens=10000000000,
        eval_interval_tokens=20000000,
        save_every=1000000000,
        lr_decay_tokens=1000000000,
    ),
    PaperRun(
        name="s_full_attention_10b_seed0",
        architecture="standard",
        tags=("s", "full_attention", "10b", "seed_sweep", "seed0"),
        max_tokens=10000000000,
        eval_interval_tokens=20000000,
        save_every=1000000000,
        seed=0,
        lr_decay_tokens=1000000000,
    ),
    PaperRun(
        name="s_full_attention_10b_seed1",
        architecture="standard",
        tags=("s", "full_attention", "10b", "seed_sweep", "seed1"),
        max_tokens=10000000000,
        eval_interval_tokens=20000000,
        save_every=1000000000,
        seed=1,
        lr_decay_tokens=1000000000,
    ),
    PaperRun(
        name="s_full_attention_20b",
        architecture="standard",
        tags=("s", "full_attention", "20b"),
    ),
    PaperRun(
        name="s_full_attention_40b",
        architecture="standard",
        tags=("s", "full_attention", "40b"),
        max_tokens=40000000000,
        lr_decay_tokens=4000000000,
    ),
    PaperRun(
        name="s_reverse_sps_w0_20b",
        architecture="reverse_sps",
        tags=(
            "s",
            "reverse_sps",
            "sliding_predict_tokens",
            "deterministic",
            "w0",
            "20b",
            "triton",
        ),
        window_size=0,
    ),
    PaperRun(
        name="s_reverse_sps_w16_20b",
        architecture="reverse_sps",
        tags=(
            "s",
            "reverse_sps",
            "sliding_predict_tokens",
            "deterministic",
            "w16",
            "20b",
            "triton",
        ),
        window_size=16,
    ),
    PaperRun(
        name="s_reverse_sps_w256_20b",
        architecture="reverse_sps",
        tags=(
            "s",
            "reverse_sps",
            "sliding_predict_tokens",
            "deterministic",
            "w256",
            "20b",
            "triton",
        ),
        window_size=256,
    ),
    PaperRun(
        name="s_reverse_sps_w64_20b",
        architecture="reverse_sps",
        tags=(
            "s",
            "reverse_sps",
            "sliding_predict_tokens",
            "deterministic",
            "w64",
            "20b",
            "triton",
        ),
        window_size=64,
    ),
    PaperRun(
        name="s_sps_w0_20b",
        architecture="sps",
        tags=(
            "s",
            "sps",
            "sliding_predict_tokens",
            "deterministic",
            "w0",
            "20b",
            "triton",
        ),
        window_size=0,
    ),
    PaperRun(
        name="s_sps_w16_20b",
        architecture="sps",
        tags=(
            "s",
            "sps",
            "sliding_predict_tokens",
            "deterministic",
            "w16",
            "20b",
            "triton",
        ),
        window_size=16,
    ),
    PaperRun(
        name="s_sps_w256_20b",
        architecture="sps",
        tags=(
            "s",
            "sps",
            "sliding_predict_tokens",
            "deterministic",
            "w256",
            "20b",
            "triton",
        ),
        window_size=256,
    ),
    PaperRun(
        name="s_sps_w4096_10b",
        architecture="sps",
        tags=(
            "s",
            "sps",
            "sliding_predict_tokens",
            "deterministic",
            "w4096",
            "10b",
            "triton",
        ),
        max_tokens=10000000000,
        eval_interval_tokens=20000000,
        save_every=1000000000,
        lr_decay_tokens=1000000000,
        window_size=4096,
    ),
    PaperRun(
        name="s_sps_w4096_10b_seed0",
        architecture="sps",
        tags=(
            "s",
            "sps",
            "sliding_predict_tokens",
            "deterministic",
            "w4096",
            "10b",
            "triton",
            "seed_sweep",
            "seed0",
        ),
        max_tokens=10000000000,
        eval_interval_tokens=20000000,
        save_every=1000000000,
        seed=0,
        lr_decay_tokens=1000000000,
        window_size=4096,
    ),
    PaperRun(
        name="s_sps_w4096_10b_seed1",
        architecture="sps",
        tags=(
            "s",
            "sps",
            "sliding_predict_tokens",
            "deterministic",
            "w4096",
            "10b",
            "triton",
            "seed_sweep",
            "seed1",
        ),
        max_tokens=10000000000,
        eval_interval_tokens=20000000,
        save_every=1000000000,
        seed=1,
        lr_decay_tokens=1000000000,
        window_size=4096,
    ),
    PaperRun(
        name="s_sps_w4096_20b",
        architecture="sps",
        tags=(
            "s",
            "sps",
            "sliding_predict_tokens",
            "deterministic",
            "w4096",
            "20b",
            "triton",
        ),
        window_size=4096,
    ),
    PaperRun(
        name="s_sps_w64_10b",
        architecture="sps",
        tags=(
            "s",
            "sps",
            "sliding_predict_tokens",
            "deterministic",
            "w64",
            "10b",
            "triton",
        ),
        max_tokens=10000000000,
        eval_interval_tokens=20000000,
        save_every=1000000000,
        lr_decay_tokens=1000000000,
        window_size=64,
    ),
    PaperRun(
        name="s_sps_w64_10b_seed0",
        architecture="sps",
        tags=(
            "s",
            "sps",
            "sliding_predict_tokens",
            "deterministic",
            "w64",
            "10b",
            "triton",
            "seed_sweep",
            "seed0",
        ),
        max_tokens=10000000000,
        eval_interval_tokens=20000000,
        save_every=1000000000,
        seed=0,
        lr_decay_tokens=1000000000,
        window_size=64,
    ),
    PaperRun(
        name="s_sps_w64_10b_seed1",
        architecture="sps",
        tags=(
            "s",
            "sps",
            "sliding_predict_tokens",
            "deterministic",
            "w64",
            "10b",
            "triton",
            "seed_sweep",
            "seed1",
        ),
        max_tokens=10000000000,
        eval_interval_tokens=20000000,
        save_every=1000000000,
        seed=1,
        lr_decay_tokens=1000000000,
        window_size=64,
    ),
    PaperRun(
        name="s_sps_w64_20b",
        architecture="sps",
        tags=(
            "s",
            "sps",
            "sliding_predict_tokens",
            "deterministic",
            "w64",
            "20b",
            "triton",
        ),
        window_size=64,
    ),
    XLRun(
        name="xl_delayed_state_w64_20b",
        architecture="delayed_state",
        tags=(
            "xl",
            "delayed_state",
            "sliding_predict_tokens",
            "deterministic",
            "w64",
            "20b",
            "triton",
        ),
        warmup_tokens=100000000,
        window_size=64,
    ),
    XLRun(
        name="xl_full_attention_20b",
        architecture="standard",
        tags=("xl", "full_attention", "20b", "triton"),
        warmup_tokens=100000000,
    ),
    XLRun(
        name="xl_sps_w4096_20b",
        architecture="sps",
        tags=("xl", "sps", "deterministic", "w4096", "20b", "triton"),
        warmup_tokens=100000000,
        window_size=4096,
    ),
    XLRun(
        name="xl_sps_w64_20b",
        architecture="sps",
        tags=("xl", "sps", "deterministic", "w64", "20b", "triton"),
        warmup_tokens=100000000,
        window_size=64,
    ),
    XSRun(
        name="xs_delayed_state_w64_20b",
        architecture="delayed_state",
        tags=(
            "xs",
            "delayed_state",
            "sliding_predict_tokens",
            "deterministic",
            "w64",
            "20b",
            "triton",
        ),
        window_size=64,
    ),
    XSRun(
        name="xs_full_attention_20b",
        architecture="standard",
        tags=("xs", "full_attention", "20b"),
    ),
    XSRun(
        name="xs_full_attention_40b",
        architecture="standard",
        tags=("xs", "full_attention", "40b"),
        max_tokens=40000000000,
        lr_decay_tokens=4000000000,
    ),
    XSRun(
        name="xs_sps_w4096_20b",
        architecture="sps",
        tags=(
            "xs",
            "sps",
            "sliding_predict_tokens",
            "deterministic",
            "w4096",
            "20b",
            "triton",
        ),
        window_size=4096,
    ),
    XSRun(
        name="xs_sps_w64_20b",
        architecture="sps",
        tags=(
            "xs",
            "sps",
            "sliding_predict_tokens",
            "deterministic",
            "w64",
            "20b",
            "triton",
        ),
        window_size=64,
    ),
]

recipes = {run.name: run for run in experiment_configs}
