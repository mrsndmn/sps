"""Token-matched Standard and zero-window SPS experiments at S scale."""

from dataclasses import dataclass, replace


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


@dataclass(frozen=True)
class Pilot:
    name: str
    recipe: str
    training: Training
    reference_name: str | None = None
    num_gpus: int = 4
    trainable: bool = True

    def config(self, data_root, *, probe=False):
        t = self.training
        step = t.global_batch_size * t.block_size
        if self.recipe not in {"s_sps_w0_20b", "s_full_attention_20b"}:
            raise ValueError(f"Unknown pilot architecture: {self.recipe}")
        sps = self.recipe == "s_sps_w0_20b"
        model = "modeling.models.sps" if sps else "modeling.models.full_attention_model"
        dimensions = dict(
            block_size=t.block_size,
            vocab_size=50304,
            n_layer=12,
            n_head=12,
            hidden_size=768,
            intermediate_size=2304,
            norm_eps=1e-6,
            dropout=0.0,
            bias=False,
            eos_token_id=50256,
            pad_token_id=50303,
            warp_specialize=False,
        )
        if sps:
            dimensions.update(
                predict_token_id=50257, window_size=0, enable_triton_attention=True
            )
        else:
            dimensions.update(use_triton_full_attention=True)
        tags = (
            [
                "s",
                "sps",
                "sliding_predict_tokens",
                "deterministic",
                "w0",
                "20b",
                "triton",
            ]
            if sps
            else ["s", "full_attention", "20b"]
        )
        return dict(
            model=dict(
                _target_=model + (".SPSModel" if sps else ".Model"),
                config=dict(
                    _target_=model + (".SPSConfig" if sps else ".ModelConfig"),
                    **dimensions,
                ),
            ),
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
                min_lr=6e-5,
            ),
            optimizer=dict(
                weight_decay=0.1,
                learning_rate=6e-4,
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
    training = Training(
        max_tokens=1_000_341_504,
        warmup_tokens=10_000_000,
        decay_tokens=100_000_000,
        eval_interval=50_000_000,
        save_interval=250_000_000,
        rolling_interval=100_000_000,
    )
    standard = Pilot("s_standard_pilot_1b", "s_full_attention_20b", training)
    return [
        standard,
        replace(standard, name="s_sps_w0_pilot_1b", recipe="s_sps_w0_20b"),
    ]


def zero_window_longer_training():
    training = Training(
        max_tokens=10_003_415_040,
        warmup_tokens=100_000_000,
        decay_tokens=1_000_000_000,
        eval_interval=250_000_000,
        save_interval=1_000_000_000,
        rolling_interval=250_000_000,
    )
    return [
        replace(
            run,
            name=run.name.replace("pilot_1b", "10b"),
            training=training,
            reference_name=run.name,
        )
        for run in zero_window_pilot()
    ]


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
