"""Dependency-free dry routing for the AR conversion suite."""
import json
import os
import subprocess
import sys


def test_dry_routing():
    env = dict(os.environ, SPS_EXPERIMENT_SUITE='ar-sps')
    for entry, args in [('run_train_jobs.py', ['--phase', 'prepare']),
                        ('run_train_jobs.py', []), ('run_eval.py', [])]:
        result = subprocess.run([sys.executable, entry, '--dry', *args], env=env,
                                capture_output=True, text=True, check=True)
        assert 'scripts.ar_sps_run' in result.stdout
        assert ('a100.1gpu' if entry == 'run_eval.py' else 'a100.4gpu') in result.stdout
        assert '"jobs": []' in result.stdout
