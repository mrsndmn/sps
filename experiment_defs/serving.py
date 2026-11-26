"""Expose the existing vLLM and draft-sweep arm definitions to the launchers."""

from experiment_defs.standalone import job, load


def serving(family):
    module = load(family)
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
        collect = load("serving_benchmark").selected
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
                config=load("vllm_sps_config"),
            )
