"""Expose the existing vLLM and draft-sweep arm definitions to the launchers."""

from orchestration.named.standalone import job, load


def shared_training_settings(source):
    fields = ("CONTEXT", "EMBED_LR", "FULL_LR", "BETAS", "WEIGHT_DECAY", "CLIP")
    return {field: getattr(source, field) for field in fields}


def serving(family):
    module = load("workloads.serving." + family)
    if family in {"draft_sweep", "serving_graphs"}:
        module.configure()
    if family == "draft_sweep":
        phases = module.PHASES
        collect = module.candidates
    elif family == "vllm_sps_run":
        phases = ("train", "probe")
        collect = module.c.collect_experiments
    else:
        phases = ("audit", "eval")
        collect = load("workloads.serving.serving_benchmark").selected
    config = load("experiments.vllm_sps_config")
    for phase in phases:
        for run in collect(phase):
            yield job(
                module,
                phase,
                run,
                "--phase",
                phase,
                "--filter",
                run.name,
                config=config,
                parameters=shared_training_settings(config.source)
                if phase in {"train", "probe"}
                else None,
            )
