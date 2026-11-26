"""Map a family ID to its lightweight experiment definitions."""

from importlib import import_module

from orchestration.suite_registry import WORKER_SCRIPTS

FAMILIES = {
    "paper-sps": "paper_reproduction",
    **dict.fromkeys(WORKER_SCRIPTS, "shared"),
    "qwen3-base-long-ce": "base50",
    "qwen3-base-sps-adapters": "qwen_adapter_run",
    "qwen3-base-sps-adapters-mlp": "qwen_adapter_run",
    "qwen3-instruct-cosine": "sched_config",
    "qwen3-instruct-extended": "sched_config",
    "qwen3-instruct-parallel-ifeval": "ifeval_config",
    "qwen3-instruct-real-readout-mtp": "real_readout_run",
    "instruct-response-provenance": "provenance_prepare",
    "paper-evidence-followup": "sampling_audit",
    "vllm-sps-qwen3": "vllm_sps_config",
    "vllm-sps-benchmark": "serving_benchmark",
    "vllm-sps-cudagraphs": "serving_benchmark",
    "vllm-sps-draft-sweep": "draft_sweep",
    "qwen4b-drafter": "qwen4b_drafter",
    "dflash-mask-ablation": "dflash_mask_ablation",
    "dflash-reproduction": "dflash_repro_train:HardDFlash",
    "dflash-soft-kd": "dflash_kd_train:SoftKD",
    "qwen3-teacher-cap-probe": "teacher_cap_probe",
    "qwen35-unloop-cap-compare": "qwen35_cap_compare",
}


def family_definition(family):
    module_name, _, class_name = FAMILIES[family].partition(":")
    module = import_module("experiments." + module_name)
    return getattr(module, class_name) if class_name else module


def family_runs(family):
    definition = family_definition(family)
    runs = {}
    for run in definition.launches(family):
        key = (run.name, run.phase)
        if key in runs:
            raise ValueError(f"Duplicate {run.phase} run {family}/{run.name}")
        runs[key] = run
    return runs
