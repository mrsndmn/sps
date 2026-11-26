import ast
import os
import subprocess
import sys
from dataclasses import is_dataclass
from importlib import import_module
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_scripts_are_entry_points_only():
    assert not list((REPO / "scripts").rglob("__init__.py"))
    for root in ("src", "scripts", "tests"):
        for path in (REPO / root).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    modules = [node.module]
                else:
                    continue
                assert all(module.split(".")[0] != "scripts" for module in modules), path


def test_canonical_experiment_definitions_are_dataclasses():
    for name in ("ar_sps_config", "instruct_config", "mmlu_config", "vllm_sps_config"):
        config = import_module(f"experiments.{name}")
        assert all(is_dataclass(exp) for exp in config.experiment_configs)


def test_worker_scripts_are_directly_executable():
    for path, option in (
        ("scripts/readout/real_readout_gsm8k.py", "--worker"),
        ("scripts/serving/vllm_sps_run.py", "--phase"),
    ):
        result = subprocess.check_output([sys.executable, path, "--help"], text=True, cwd=REPO)
        assert option in result


@pytest.mark.parametrize(
    "path,dependency",
    [
        ("scripts/benchmark/benchmark_generation_speed.py", "torch"),
        ("scripts/figures/plot_accuracy_gain_by_scale.py", "matplotlib"),
    ],
)
def test_paper_scripts_load_this_checkout(path, dependency):
    pytest.importorskip(dependency)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    output = subprocess.check_output(
        [sys.executable, path, "--help"], cwd=REPO, env=env, text=True
    )
    assert output.startswith("usage:")
