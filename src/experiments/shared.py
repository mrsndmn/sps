"""Named phases of families that use the shared MLSpace launcher."""

from importlib import import_module

from orchestration.suite_registry import AR_SPS_SUITES, EVAL_ONLY_SUITES, MMLU_SUITES
from .launch import Launch


def records(family):
    if family in AR_SPS_SUITES:
        from .ar_sps import configuration

        return configuration(family).experiment_configs
    if family in MMLU_SUITES:
        from .mmlu import experiments

        return experiments(family)
    if family in {"mtp", "mtp-unfreeze", "mtp-parallel"}:
        from .mtp_design import configuration

        return configuration(family).experiment_configs
    module = {
        "1b": "pilot_1b",
        "10b": "pilot_10b",
        "spec-baselines": "spec_baselines",
        "mtp-unified": "mtp_unified",
        "ar-sps-qwen3-1.7b-instruct": "instruct_config",
    }[family]
    return import_module("experiments." + module).experiment_configs


def launch_environment(family):
    return {"SPS_EXPERIMENT_SUITE": family}


def shared_launch(name, phase):
    return Launch(
        name,
        phase,
        "scripts/run_train_jobs.py",
        ("--phase", phase, "--exact", name),
        options=("profile", "author_name", "telegram_nick"),
    )


def launches(family):
    if family not in EVAL_ONLY_SUITES:
        yield shared_launch("data-and-probes", "prepare")
    for row in records(family):
        yield shared_launch(row.name, "eval")
        if family not in EVAL_ONLY_SUITES and row.trainable:
            yield shared_launch(row.name, "train")
