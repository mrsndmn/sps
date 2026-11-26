"""Adapters for job protocols that already own checkpoint selection or reservations."""

from functools import partial
from orchestration.named.distillation import dflash, qwen4b
from orchestration.named.instruction import ifeval, schedules
from orchestration.named.serving import serving, shared_training_settings
from orchestration.named.standalone import builds, job, load
from orchestration.suite_registry import INSTRUCT_SUITE


def adapters():
    module = load("workloads.adaptation.qwen_adapter_run")
    for run in module.experiment_configs:
        yield job(
            module,
            "train",
            run,
            "train",
            "--filter",
            run.name,
            output_dir=module.directory(run),
            config=load("experiments.ar_sps_config"),
        )
    yield job(module, "eval", "winner", "eval", output_dir=module.winner_output(), config=load("experiments.ar_sps_config"))


def base50():
    module = load("workloads.adaptation.base50_run")
    yield job(
        module,
        "train",
        module.c.experiment_configs[0],
        config=(module.c, module.config),
        output_dir=module.e.RUN_ROOT / "out" / module.c.experiment_configs[0].name,
    )


def readout():
    module = load("workloads.readout.real_readout_run")
    for phase in ("probe", "train"):
        yield job(
            module,
            phase,
            module.ARM,
            phase,
            config=module.c,
            output_dir=module.ROOT / "probe" if phase == "probe" else module.OUT,
            parameters=shared_training_settings(module.c.source),
        )
    for workload in ("eval", "mmlu", "gsm8k", "humaneval", "serving"):
        module = load("workloads.readout.real_readout_" + workload)
        for exp in module.experiment_configs:
            selector = (
                ("--arm", exp.arm) if workload == "serving" else ("--filter", exp.name)
            )
            source = load("experiments.instruct_config")
            config = (source, load("experiments.vllm_sps_config"), module)
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
                output_dir=module.ROOT / "runs/out" / exp.name,
            )


def provenance():
    source = load("experiments.instruct_config")
    prep = load("workloads.provenance.provenance_prepare")
    yield from builds("workloads.provenance.provenance_prepare", "prepare", config=(source, prep))
    yield from builds("workloads.provenance.provenance_train", "train", "--arm", config=(source, prep))
    evaluator = load("workloads.provenance.provenance_evaluate")
    for run in evaluator.experiment_configs:
        yield job(
            evaluator,
            "eval",
            run,
            "--filter",
            run.name,
            config=(source, evaluator.ic),
            output_dir=evaluator.output_dir(run),
            parameters={
                "generation": source.GENERATION[run.mode],
                "preparation_generation": prep.GENERATION,
            },
        )


def sampling():
    shared = (load("experiments.instruct_config"), load("experiments.vllm_sps_config"), load("workloads.serving.sampling_audit"))
    for suffix in ("audit", "trace", "precision", "cache_trace", "backbone_precision"):
        module = load("workloads.serving.sampling_" + suffix)
        conditions = [vars(run) for run in module.experiment_configs]
        for run in getattr(module, "job_configs", module.experiment_configs):
            yield job(
                module,
                "eval",
                run,
                config=shared,
                parameters={"conditions": conditions},
                output_dir=(module.output_dir() if suffix in {"audit", "trace"} else module.root()),
            )


def mask_ablation():
    module = load("workloads.distillation.dflash_mask_ablation")
    yield job(
        module,
        "eval",
        "paired-masks",
        submit_flag="--submit",
        output_dir=module.result_path().parent,
        parameters={"conditions": [vars(run) for run in module.experiment_configs]},
    )


def cap_probe():
    from orchestration.named import cap_jobs

    module = load("workloads.evaluation.teacher_cap_probe")
    (probe,) = module.experiment_configs
    yield job(
        cap_jobs,
        "eval",
        probe,
        "teacher",
        name="continuations",
        output_dir=module.ROOT / cap_jobs.teacher_config.OUTPUT_NAME,
        config=(module.c, module, cap_jobs.teacher_config),
    )


def cap_compare():
    from orchestration.named import cap_jobs

    module = load("workloads.evaluation.qwen35_cap_compare")
    for arm, run in module.experiment_configs.items():
        yield job(
            cap_jobs,
            "eval",
            run,
            "compare",
            arm,
            config=(module.c, module, cap_jobs.comparison_config),
            output_dir=module.folder(arm),
            parameters={"MODEL": run.model, "REVISION": run.revision},
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
            "SPS_ADAPTER_SCOPE": (
                "attention-mlp" if family.endswith("-mlp") else "attention"
            ),
        }
    return {"SPS_EXPERIMENT_SUITE": INSTRUCT_SUITE}
