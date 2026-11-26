"""Token-matched Standard and zero-window SPS experiments at S scale."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Training:
    max_tokens: int
    warmup_tokens: int
    decay_tokens: int
    eval_interval: int
    save_interval: int
    rolling_interval: int
    micro_batch_size: int = 6
    global_batch_size: int = 96
    block_size: int = 4096
    chunk_size_units: int = 1024
    probe_steps: int = 16
    seed: int = 1337
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1


@dataclass(frozen=True)
class Model:
    n_layer: int = 12
    n_head: int = 12
    hidden_size: int = 768
    intermediate_size: int = 2304
    window: int | None = None

    def config(self, block_size):
        sps = self.window is not None
        model = "modeling.models.sps" if sps else "modeling.models.full_attention_model"
        dimensions = dict(
            block_size=block_size,
            vocab_size=50304,
            n_layer=self.n_layer,
            n_head=self.n_head,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            norm_eps=1e-6,
            dropout=0.0,
            bias=False,
            eos_token_id=50256,
            pad_token_id=50303,
            warp_specialize=False,
        )
        if sps:
            dimensions.update(
                predict_token_id=50257,
                window_size=self.window,
                enable_triton_attention=True,
            )
        else:
            dimensions.update(use_triton_full_attention=True)
        return dict(
            _target_=model + (".SPSModel" if sps else ".Model"),
            config=dict(
                _target_=model + (".SPSConfig" if sps else ".ModelConfig"), **dimensions
            ),
        )


@dataclass(frozen=True)
class Pilot:
    name: str
    model: Model
    training: Training
    output_dir: Path
    eval_output_dir: Path
    reference_name: str | None = None
    num_gpus: int = 4
    trainable: bool = True

    @property
    def recipe(self):
        # Original Hydra name retained for legacy callers and experiment metadata.
        return (
            "s_full_attention_20b"
            if self.model.window is None
            else f"s_sps_w{self.model.window}_20b"
        )

    def arguments(self, data_root, *, probe=False):
        # Flatten the resolved groups into the typed CLI namespace.
        cfg = self.config(data_root, probe=probe)
        model = dict(cfg["model"]["config"])
        del model["_target_"]
        return dict(**model, architecture="standard" if self.model.window is None else "sps",
                    **cfg["experiment"], **cfg["data"], **cfg["training"], **cfg["scheduler"],
                    **cfg["optimizer"], **cfg["system"], **cfg["logging"],
                    total_tokens=cfg["eval"]["total_tokens"], validation_seed=cfg["eval"]["seed"])

    def config(self, data_root, *, probe=False):
        t = self.training
        step = t.global_batch_size * t.block_size
        tags = (
            [
                "s",
                "sps",
                "sliding_predict_tokens",
                "deterministic",
                f"w{self.model.window}",
                "20b",
                "triton",
            ]
            if self.model.window is not None
            else ["s", "full_attention", "20b"]
        )
        return dict(
            model=self.model.config(t.block_size),
            experiment=dict(name=self.name, tags=tags),
            data=dict(dataset="fineweb-edu"),
            training=dict(
                micro_batch_size=t.micro_batch_size,
                global_batch_size=t.global_batch_size,
                max_tokens=t.probe_steps * step if probe else t.max_tokens,
                eval_total_tokens=step if probe else 1048576,
                eval_interval_tokens=t.max_tokens if probe else t.eval_interval,
                eval_only=False,
                skip_first_eval=probe,
                val_max_tokens=None,
                log_interval=1,
                always_save_checkpoint=True,
                save_every=t.max_tokens if probe else t.save_interval,
                rolling_save_every=t.max_tokens if probe else t.rolling_interval,
                sampler_type="fixed_random_chunk",
                chunk_shuffle_seed=t.seed,
                seed=t.seed,
                chunk_size_units=t.chunk_size_units,
                init_from="scratch" if probe else "resume",
                resume_checkpoint=None,
            ),
            scheduler=dict(
                decay_lr=True,
                warmup_tokens=step if probe else t.warmup_tokens,
                lr_decay_tokens=step if probe else t.decay_tokens,
                min_lr=t.min_lr,
            ),
            optimizer=dict(
                weight_decay=t.weight_decay,
                learning_rate=t.learning_rate,
                beta1=0.9,
                beta2=0.95,
                grad_clip=1.0,
            ),
            system=dict(
                backend="nccl",
                dtype="bfloat16",
                device="cuda",
                compile=False,
                deterministic=False,
                data_root=str(data_root),
            ),
            logging=dict(
                wandb_log=False,
                wandb_project="sps",
                wandb_entity=None,
                wandb_run_name=self.name,
                wandb_dir=str(data_root) + "/wandb",
            ),
            eval=dict(total_tokens=10000000, seed=100),
        )


def zero_window_pilot():
    from experiments.pilot_1b import experiment_configs
    return list(experiment_configs)


def zero_window_longer_training():
    from experiments.pilot_10b import experiment_configs
    return list(experiment_configs)


DATASET_REPO = "HuggingFaceFW/fineweb-edu"
DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


# Three complete sample-10BT shards supply roughly two billion source tokens.
def data_files(suite):
    if suite == "10b":
        return tuple(f"sample/100BT/{i:03d}_00000.parquet" for i in range(14)) + (
            "sample/100BT/000_00001.parquet",
            "sample/100BT/001_00001.parquet",
        )
    return tuple(f"sample/10BT/{i:03d}_00000.parquet" for i in range(3))


EXISTING_HF_CACHES = (
    Path("/home/jovyan/.cache/huggingface/hub"),
    Path("/mnt/virtual_ai0001053-00054_SR004-nfs2/.cache/huggingface/hub"),
    Path("/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/hf-cache/hub"),
    Path("/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/HF_CACHE_NFS1/hub"),
)
TASKS = ("arc_easy", "hellaswag", "piqa", "sciq", "lambada_openai")

MAX_HOURS = 24
VAL_FRACTION = 0.01
VAL_SEED = 2357


def min_free_bytes(suite):
    return (120 if suite == "10b" else 40) * 10**9


# The long run has a complete prior run for calibration rather than only a
# short probe. Keep a margin plus one hour for startup and checkpoint I/O.
def runtime_safety_factor(suite):
    return 1.15 if suite == "10b" else 1.5
