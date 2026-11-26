import subprocess
import sys
from importlib import import_module


def test_flat_and_grouped_imports_share_state():
    for original, grouped in [('real_readout_run', 'readout.real_readout_run'),
                              ('vllm_sps_config', 'serving.vllm_sps_config'),
                              ('draft_sweep_train', 'serving.draft_sweep_train'),
                              ('mmlu_results', 'evaluation.mmlu_results')]:
        assert import_module(f'scripts.{original}') is import_module(f'scripts.{grouped}')


def test_moved_config_imports_use_the_canonical_dataclass_definitions():
    from dataclasses import is_dataclass

    for name in ('ar_sps_config', 'instruct_config', 'mmlu_config', 'vllm_sps_config'):
        config = import_module(f'src.experiments.{name}')
        assert config is import_module(f'scripts.{name}')
        assert all(is_dataclass(experiment) for experiment in config.experiment_configs)


def test_flat_and_grouped_worker_cli():
    for module in ('scripts.real_readout_gsm8k', 'scripts.readout.real_readout_gsm8k'):
        result = subprocess.check_output([sys.executable, '-m', module, '--help'], text=True)
        assert '--worker' in result
    for module in ('scripts.vllm_sps_run', 'scripts.serving.vllm_sps_run'):
        result = subprocess.check_output([sys.executable, '-m', module, '--help'], text=True)
        assert '--phase' in result
