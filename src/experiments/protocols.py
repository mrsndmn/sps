"""Argument groups consumed by the native MLSpace worker for each experiment."""

INSTRUCTION = ("instruct_config",)
SERVING = (*INSTRUCTION, "vllm_sps_config")
READOUT = (*SERVING, "real_readout_run")
SAMPLING = (*SERVING, "sampling_audit")
DRAFTER = (
    *SERVING,
    "dflash_mask_ablation",
    "qwen4b_drafter_data",
    "qwen4b_drafter_train",
)

WORKERS = {
    "base50_run": ("ar_sps_config", "base50"),
    "qwen_adapter_run": ("ar_sps_config", "qwen_adapter_run"),
    "sched_run": (*INSTRUCTION, "sched_config"),
    "extend_run": (*INSTRUCTION, "extend_config"),
    "ifeval_run": (*INSTRUCTION, "ifeval_config"),
    "real_readout_run": READOUT,
    "real_readout_eval": (*READOUT, "ifeval_config", "real_readout_eval"),
    "real_readout_gsm8k": (*READOUT, "real_readout_eval", "real_readout_gsm8k"),
    "real_readout_mmlu": (*READOUT, "real_readout_serving", "real_readout_mmlu"),
    "real_readout_humaneval": (*READOUT, "real_readout_eval", "real_readout_humaneval"),
    "real_readout_serving": (*READOUT, "real_readout_eval", "real_readout_serving"),
    "provenance_prepare": (*SERVING, "provenance_prepare"),
    "provenance_train": (*SERVING, "provenance_prepare", "provenance_train"),
    "provenance_evaluate": (
        *SERVING,
        "provenance_prepare",
        "provenance_train",
        "provenance_evaluate",
        "ifeval_config",
    ),
    "sampling_audit": SAMPLING,
    "sampling_trace": (*SAMPLING, "sampling_trace"),
    "sampling_precision": (*SAMPLING, "sampling_trace", "sampling_precision"),
    "sampling_cache_trace": (
        *SAMPLING,
        "sampling_trace",
        "sampling_precision",
        "sampling_cache_trace",
    ),
    "sampling_backbone_precision": (
        *SAMPLING,
        "sampling_trace",
        "sampling_precision",
        "sampling_backbone_precision",
    ),
    "vllm_sps_run": SERVING,
    "serving_benchmark": (*SERVING, "serving_benchmark"),
    "serving_graphs": (*SERVING, "serving_benchmark", "serving_graphs"),
    "draft_sweep": (*SERVING, "draft_sweep_train", "draft_sweep"),
    "qwen4b_drafter": (*DRAFTER, "qwen4b_drafter"),
    "dflash_mask_ablation": (*SERVING, "dflash_mask_ablation"),
    "cap_jobs": (*SERVING, "teacher_cap_probe", "qwen35_cap_compare"),
}

EVALUATION = {
    "draft_sweep": ("serving_benchmark", "serving_graphs"),
    "qwen4b_drafter": ("qwen4b_drafter_eval",),
}


def schemas(worker, phase):
    return WORKERS.get(worker, ()) + (
        EVALUATION.get(worker, ()) if phase in {"eval", "audit"} else ()
    )
