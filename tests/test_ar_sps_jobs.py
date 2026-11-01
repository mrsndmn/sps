"""Dependency-free dry routing for the AR conversion suite."""
import json
import os
import subprocess
import sys
import pytest


@pytest.mark.parametrize('suite', ['ar-sps', 'ar-sps-kd'])
def test_dry_routing(suite, tmp_path):
    env = dict(os.environ, SPS_EXPERIMENT_SUITE=suite)
    for entry, args in [('run_train_jobs.py', ['--phase', 'prepare']),
                        ('run_train_jobs.py', []), ('run_eval.py', [])]:
        # Isolate orchestration from real completed/stale experiment artifacts.
        code = ("import sys; from pathlib import Path; import experiments as e; "
                "e.ARTIFACTS_ROOT=Path(sys.argv[1]); e.RUN_ROOT=e.ARTIFACTS_ROOT/'runs'; "
                "from scripts.mlspace_launch import main; "
                "phase=sys.argv[2]; sys.argv=['launcher', '--dry', *sys.argv[3:]]; main(phase)")
        result = subprocess.run([sys.executable, '-c', code, str(tmp_path/suite),
                                 'eval' if entry == 'run_eval.py' else 'train', *args], env=env,
                                capture_output=True, text=True, check=True)
        if suite == 'ar-sps-kd' and entry == 'run_eval.py':
            assert 'Skip evaluation without trained checkpoint' in result.stdout
            continue
        assert 'scripts.ar_sps_run' in result.stdout
        assert ('a100.1gpu' if entry == 'run_eval.py' else 'a100.4gpu') in result.stdout
        assert '"jobs": []' in result.stdout


def test_real_static_evaluation_requires_preparation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest
    from scripts import mlspace_launch as launch
    monkeypatch.setattr(launch, 'SUITE', 'ar-sps')
    monkeypatch.setattr(launch, 'ARTIFACTS_ROOT', tmp_path)
    monkeypatch.setattr(launch, 'collect_experiments', lambda _: [SimpleNamespace(name='source', trainable=False)])
    monkeypatch.setattr(launch, 'output_dir', lambda _: tmp_path/'source')
    args = SimpleNamespace(phase='eval', dry=False, filter=None)
    with pytest.raises(RuntimeError, match='prepare'):
        launch.submit(args)
    (tmp_path/'prepared.json').write_text(json.dumps({'design_signature': 'obsolete'}))
    monkeypatch.setattr(launch, 'design_signature', lambda: 'current')
    with pytest.raises(RuntimeError, match='Stale'):
        launch.submit(args)
