import subprocess
import sys
from importlib import import_module


def test_flat_and_grouped_imports_share_state():
    for original, grouped in [('real_readout_run', 'readout.real_readout_run'),
                              ('mmlu_results', 'evaluation.mmlu_results')]:
        assert import_module(f'scripts.{original}') is import_module(f'scripts.{grouped}')


def test_flat_and_grouped_worker_cli():
    for module in ('scripts.real_readout_gsm8k', 'scripts.readout.real_readout_gsm8k'):
        result = subprocess.check_output([sys.executable, '-m', module, '--help'], text=True)
        assert '--worker' in result
