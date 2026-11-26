"""Adapters for job protocols that already own checkpoint selection or reservations."""

from functools import partial
from experiment_defs.distillation import dflash, qwen4b
from experiment_defs.instruction import ifeval, schedules
from experiment_defs.serving import serving
from experiment_defs.standalone import builds, job, load
from scripts.suite_registry import INSTRUCT_SUITE


def adapters():
    module = load("qwen_adapter_run")
    for run in module.experiment_configs:
        yield job(module, "train", run, "train", "--filter", run.name)
    yield job(module, "eval", "winner", "eval")


def base50():
    module = load("base50_run")
    yield job(
        module,
        "train",
        module.c.experiment_configs[0],
        config=(module.c, module.config),
    )


def readout():
    module = load("real_readout_run")
    for phase in ("probe", "train"):
        yield job(module, phase, module.ARM, phase, config=module.c)
    for workload in ("eval", "mmlu", "gsm8k", "humaneval", "likelihood", "serving"):
        module = load("real_readout_" + workload)
        for exp in module.experiment_configs:
            selector = (
                ("--arm", exp.arm)
                if workload in {"serving", "likelihood"}
                else ("--filter", exp.name)
            )
            source = load("instruct_config")
            config = (source, load("vllm_sps_config"), module)
            effective = {}
            if exp.mode is not None:
                modes = getattr(module, "SETTINGS", source.GENERATION)
                effective["generation"] = modes[exp.mode]
            yield job(
                module,
                "eval",
                exp,
                *selector,
                name=f"{workload}-{exp.name}",
                config=config,
                parameters=effective,
            )


def provenance():
    yield from builds("provenance_prepare", "prepare")
    yield from builds("provenance_train", "train", "--arm")
    yield from builds("provenance_evaluate", "eval", "--filter")


def sampling():
    for suffix in ("audit", "trace", "precision", "cache_trace", "backbone_precision"):
        yield from builds("sampling_" + suffix, "eval")


def mask_ablation():
    module = load("dflash_mask_ablation")
    yield job(
        module,
        "eval",
        "paired-masks",
        submit_flag="--submit",
        parameters={"conditions": [vars(run) for run in module.experiment_configs]},
    )


def cap_probe():
    from experiment_defs import cap_jobs

    module = load("teacher_cap_probe")
    (probe,) = module.experiment_configs
    yield job(
        cap_jobs,
        "eval",
        probe,
        "teacher",
        name="continuations",
        config=(module.c, module),
    )


def cap_compare():
    from experiment_defs import cap_jobs

    module = load("qwen35_cap_compare")
    for arm in module.experiment_configs:
        model, revision = module.experiment_configs[arm]
        yield job(
            cap_jobs,
            "eval",
            arm,
            "compare",
            arm,
            config=(module.c, module),
            parameters={"MODEL": model, "REVISION": revision},
        )


FAMILIES = {
    "qwen3-base-long-ce": base50,
    "qwen3-base-sps-adapters": adapters,
    "qwen3-base-sps-adapters-mlp": adapters,
    "qwen3-instruct-cosine": partial(schedules, False),
    "qwen3-instruct-extended": partial(schedules, True),
    "qwen3-instruct-parallel-ifeval": ifeval,
    "qwen3-instruct-real-readout-mtp": readout,
    "instruct-response-provenance": provenance,
    "paper-evidence-followup": sampling,
    "vllm-sps-qwen3": partial(serving, "vllm_sps_run"),
    "vllm-sps-benchmark": partial(serving, "serving_benchmark"),
    "vllm-sps-cudagraphs": partial(serving, "serving_graphs"),
    "vllm-sps-draft-sweep": partial(serving, "draft_sweep"),
    "qwen4b-drafter": qwen4b,
    "dflash-mask-ablation": mask_ablation,
    "dflash-reproduction": partial(dflash, False),
    "dflash-soft-kd": partial(dflash, True),
    "qwen3-teacher-cap-probe": cap_probe,
    "qwen35-unloop-cap-compare": cap_compare,
}


def environment(family):
    if family.startswith("qwen3-base-"):
        return {
            "SPS_EXPERIMENT_SUITE": "ar-sps-qwen3-1.7b",
            "SPS_ADAPTER_SCOPE": "attention-mlp"
            if family.endswith("-mlp")
            else "attention",
        }
    return {"SPS_EXPERIMENT_SUITE": INSTRUCT_SUITE}
