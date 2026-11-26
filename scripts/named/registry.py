"""Catalog of named experiments. Family IDs disambiguate historical run names."""

from dataclasses import asdict
from functools import partial

from src.experiments import (
    ar_sps,
    instruct_config,
    mmlu,
    mtp_design,
    mtp_unified,
    spec_baselines,
)
from src.experiments.pilot import zero_window_longer_training, zero_window_pilot
from scripts.suite_registry import AR_SPS_SUITES, INSTRUCT_SUITE, MMLU_SUITES


def settings(module):
    return {key: value for key, value in vars(module).items() if key.isupper()}


CONFIGURATIONS = {
    **{suite: ar_sps.configuration for suite in AR_SPS_SUITES},
    **{
        suite: mtp_design.configuration
        for suite in ("mtp", "mtp-parallel", "mtp-unfreeze")
    },
}


def configured_runs(build, family):
    return build(family).experiment_configs


FAMILIES = {
    "1b": zero_window_pilot,
    "10b": zero_window_longer_training,
    "spec-baselines": lambda: spec_baselines.experiment_configs,
    "mtp-unified": lambda: mtp_unified.experiment_configs,
    INSTRUCT_SUITE: lambda: instruct_config.experiment_configs,
    **{suite: partial(mmlu.experiments, suite) for suite in sorted(MMLU_SUITES)},
    **{
        suite: partial(configured_runs, build, suite)
        for suite, build in sorted(CONFIGURATIONS.items())
    },
}


def effective_settings(family, run, root, phase):
    if family in {"1b", "10b"}:
        from scripts.experiment_suite import (
            TASKS,
            DATASET_REPO,
            DATASET_REVISION,
            DATA_FILES,
            eval_command,
            train_command,
        )

        probe = phase == "prepare"
        result = run.config(root, probe=probe)
        result["data_source"] = dict(
            repo=DATASET_REPO, revision=DATASET_REVISION, files=DATA_FILES
        )
        if phase == "eval":
            result["worker_commands"] = [eval_command(run, task) for task in TASKS]
        else:
            result["worker_commands"] = [train_command(run, probe=probe)]
        return result
    if family in CONFIGURATIONS:
        common = settings(CONFIGURATIONS[family](family))
        if family in {"mtp", "mtp-parallel", "mtp-unfreeze"}:
            common.update(
                BLOCK_SIZE=common["MTP_CONTEXT"],
                GLOBAL_BATCH_SIZE=common["MTP_GLOBAL_BATCH"],
                TOKENS_PER_STEP=common["MTP_STEP_TOKENS"],
                TRAIN_TOKENS=common["MTP_TRAIN_STEPS"] * common["MTP_STEP_TOKENS"],
                PROBE_STEPS=common["MTP_PROBE_STEPS"],
            )
            if common["MTP_UNFREEZE"]:
                common["MAX_HOURS"] = common["MTP_UNFREEZE_MAX_HOURS"]
    elif family in {"spec-baselines", "mtp-unified"}:
        common = {
            **settings(mtp_design.configuration(family)),
            **settings(spec_baselines if family == "spec-baselines" else mtp_unified),
        }
        if family == "spec-baselines":
            common.update(
                MAX_HOURS=spec_baselines.BASELINE_MAX_HOURS,
                TRAIN_TOKENS=spec_baselines.BASELINE_STEPS
                * spec_baselines.BASELINE_STEP_TOKENS,
            )
    elif family in MMLU_SUITES:
        common = settings(mmlu)
    else:
        common = settings(instruct_config)
    return dict(run=asdict(run), settings=common)


def named_runs():
    runs = {}
    for family, build in FAMILIES.items():
        for run in build():
            key = f"{family}/{run.name}"
            if key in runs:
                raise ValueError(f"Duplicate experiment name: {key}")
            runs[key] = (family, run)
    return runs


def resolve(name, runs):
    if name in runs:
        return runs[name]
    candidates = [key for key in runs if key.rpartition("/")[2] == name]
    if len(candidates) == 1:
        return runs[candidates[0]]
    if candidates:
        raise ValueError(
            "Ambiguous experiment name. Select one of " + ", ".join(candidates)
        )
    raise ValueError(f"Unknown experiment {name!r}. See --list.")
