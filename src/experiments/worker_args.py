"""Typed arguments shared by the training, evaluation, and generation CLIs."""

from dataclasses import asdict, dataclass, field
from functools import wraps
from importlib import import_module
from pathlib import Path
import argparse
import sys
from typing import Literal, Optional


@dataclass
class TrainArgs:
    distributed_validation: bool = False
    micro_batch_size: int = 6
    global_batch_size: int = 96
    max_tokens: int = 10000000000
    eval_total_tokens: int = 10000000
    eval_interval_tokens: int = 20000000
    eval_only: bool = False
    skip_first_eval: bool = False
    val_max_tokens: Optional[int] = None
    log_interval: int = 1
    always_save_checkpoint: bool = True
    save_every: int = 1000000000
    rolling_save_every: int = 100000000
    sampler_type: str = "fixed_random_chunk"
    chunk_shuffle_seed: int = 1337
    seed: int = 1337
    chunk_size_units: int = 262144
    init_from: str = "resume"
    resume_checkpoint: Optional[str] = None


@dataclass
class OptimizerArgs:
    weight_decay: float = 0.1
    learning_rate: float = 0.0006
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0


@dataclass
class SchedulerArgs:
    decay_lr: bool = True
    warmup_tokens: int = 50000000
    lr_decay_tokens: int = 1000000000
    min_lr: float = 6e-05


@dataclass
class ModelArgs:
    architecture: Literal["standard", "sps", "reverse_sps", "delayed_state"] = (
        "standard"
    )
    block_size: int = 4096
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    hidden_size: int = 768
    intermediate_size: int = 2304
    norm_eps: float = 1e-6
    dropout: float = 0.0
    bias: bool = False
    eos_token_id: int = 50256
    pad_token_id: int = 50303
    use_triton_full_attention: bool = True
    warp_specialize: bool = False
    window_size: Optional[int] = None
    enable_triton_attention: bool = True
    predict_token_id: int = 50257


@dataclass
class DataArgs:
    dataset: str = "fineweb-edu"
    num_proc: Optional[int] = None
    val_split_seed: int = 42


@dataclass
class SystemArgs:
    ddp_timeout_seconds: Optional[int] = None
    backend: str = "nccl"
    dtype: str = "bfloat16"
    device: str = "cuda"
    compile: bool = True
    deterministic: bool = False
    data_root: Optional[str] = None


@dataclass
class LoggingArgs:
    wandb_log: bool = True
    wandb_project: str = "sps"
    wandb_entity: Optional[str] = None
    wandb_run_name: str = "default_run"
    wandb_dir: Optional[str] = None


@dataclass
class ValidationArgs:
    total_tokens: int = 10000000
    validation_seed: int = 100


@dataclass
class RuntimeArgs:
    name: str = "default_run"
    tags: list[str] = field(default_factory=list)
    out_dir: Optional[str] = None
    output_path: Optional[str] = None
    checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None
    apply_checkpoint_model_args: Optional[bool] = None


@dataclass
class EvaluationArgs:
    tasks: Optional[str] = None
    eval_batch_size: str = "auto"
    max_batch_size: int = 4096
    num_fewshot: int = 0
    limit: Optional[float] = None
    eval_seed: int = 42
    min_eval_seq_len: Optional[int] = None
    required_block_size: Optional[int] = None
    eval_profile: Optional[str] = None
    task_metadata_b64: Optional[str] = None
    task_config_b64: Optional[str] = None
    use_forward_efficient: bool = False


@dataclass
class GenerationArgs:
    batch_prompts: Optional[list[str]] = None
    prompt_file: Optional[str] = None
    generation_batch_size: int = 8
    interactive: bool = True
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 1.0
    top_k: Optional[int] = None
    stop_on_eos: bool = True
    truncate_left: bool = False
    generation_seed: int = 1337


GROUPS = dict(
    training=TrainArgs,
    optimizer=OptimizerArgs,
    scheduler=SchedulerArgs,
    model=ModelArgs,
    data=DataArgs,
    system=SystemArgs,
    logging=LoggingArgs,
    eval=ValidationArgs,
)
MODELS = {
    "standard": ("full_attention_model", "Model", "ModelConfig"),
    "sps": ("sps", "SPSModel", "SPSConfig"),
    "delayed_state": ("delayed_state", "DelayedStateModel", "DelayedStateConfig"),
    "reverse_sps": ("reverse_sps", "ReverseSPSModel", "ReverseSPSConfig"),
}


def instantiate(model):
    module_name, cls_name = model._target_.rsplit(".", 1)
    module = import_module(module_name)
    args = dict(model.config)
    config_name = args.pop("_target_").rsplit(".", 1)[1]
    return getattr(module, cls_name)(getattr(module, config_name)(**args))


def model_config(args):
    values = asdict(args)
    architecture = values.pop("architecture")
    module, cls, config = MODELS[architecture]
    if architecture == "standard":
        for key in ("predict_token_id", "window_size", "enable_triton_attention"):
            del values[key]
    else:
        del values["use_triton_full_attention"]
        if values["window_size"] is None:
            values["window_size"] = 16
    prefix = "modeling.models." + module + "."
    return dict(_target_=prefix + cls, config=dict(_target_=prefix + config, **values))


def cli_args(values):
    args = []
    for key, value in values.items():
        if value is None:
            continue
        args.append("--" + key)
        if isinstance(value, (list, tuple)):
            args.extend(map(str, value))
        else:
            args.append(str(value))
    return args


def parse_config(phase, argv=None):
    from omegaconf import OmegaConf
    from transformers import HfArgumentParser
    from experiments.paper import recipes

    argv = sys.argv[1:] if argv is None else argv
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--recipe", choices=recipes)
    selected, remaining = selector.parse_known_args(argv)
    extra = {"eval": (EvaluationArgs,), "generate": (GenerationArgs,)}.get(phase, ())
    parser = HfArgumentParser(
        (*GROUPS.values(), RuntimeArgs, *extra), allow_abbrev=False, parents=[selector]
    )
    if selected.recipe:
        remaining = cli_args(recipes[selected.recipe].arguments()) + remaining
    parsed = parser.parse_args_into_dataclasses(
        args=remaining, look_for_args_file=False
    )
    parsed = parsed[: len(GROUPS) + 1 + len(extra)]
    groups = {k: asdict(v) for k, v in zip(GROUPS, parsed)}
    groups["model"] = model_config(parsed[list(GROUPS).index("model")])
    groups["eval"]["seed"] = groups["eval"].pop("validation_seed")
    groups["data"] = {k: v for k, v in groups["data"].items() if v is not None}
    if groups["system"]["data_root"] and not groups["logging"]["wandb_dir"]:
        groups["logging"]["wandb_dir"] = str(
            Path(groups["system"]["data_root"]) / "wandb"
        )
    runtime = asdict(parsed[len(GROUPS)])
    groups["experiment"] = dict(name=runtime.pop("name"), tags=runtime.pop("tags"))
    groups.update({k: v for k, v in runtime.items() if v is not None})
    for obj in parsed[len(GROUPS) + 1 :]:
        groups.update(asdict(obj))
    if phase == "generate":
        groups["seed"] = groups.pop("generation_seed")
    return OmegaConf.create(groups)


def worker_cli(phase):
    def decorate(run):
        @wraps(run)
        def main():
            return run(parse_config(phase))

        return main

    return decorate
