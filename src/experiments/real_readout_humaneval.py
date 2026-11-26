"""Executable definitions for real readout humaneval."""

from . import real_readout_run as train, vllm_sps_config as c

ROOT = train.ROOT / "humaneval-plus"

PROMPTS = ROOT / "input/prompts.jsonl"

RELEASE = "v0.1.10"

URL = f"https://github.com/evalplus/humanevalplus_release/releases/download/{RELEASE}/HumanEvalPlus.jsonl.gz"

ARCHIVE_SHA = "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101"

PROMPT_SHA = "2daaf2edc5a3dc6a68316a89800b61af7d534bca5f6230a5e6b6773411576c0a"

REQUEST = (
    "Implement the Python function described below. Include its signature, all required "
    "imports and any helper functions. Return the complete solution in one Python code block."
)

SETTINGS = {
    mode: dict(settings, max_new_tokens=4096) for mode, settings in c.GENERATION.items()
}

experiment_configs = [
    c.Experiment(f"{arm}-{mode}", arm, mode) for arm in ("A", "B") for mode in SETTINGS
]

HOURS = 16
