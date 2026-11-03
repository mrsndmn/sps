"""Dependency-free instruct suite dispatch and preparation provenance gates."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

SUITE = 'ar-sps-qwen3-1.7b-instruct'


def run_code(code, *args):
    return subprocess.run([sys.executable, '-c', code, *map(str, args)],
                          env=dict(os.environ, SPS_EXPERIMENT_SUITE=SUITE),
                          capture_output=True, text=True, check=True)


@pytest.mark.parametrize('phase,count,gpus', [('prepare', 1, 4), ('train', 4, 4), ('eval', 5, 1)])
def test_instruct_dry_routing(tmp_path, phase, count, gpus):
    result = run_code('''
import sys
from pathlib import Path
import experiments as e
e.ARTIFACTS_ROOT = Path(sys.argv[1]); e.RUN_ROOT = e.ARTIFACTS_ROOT/'runs'
for exp in e.collect_experiments():
    marker = e.training_marker(exp)
    if marker is not None:
        marker.parent.mkdir(parents=True, exist_ok=True)
        if sys.argv[2] == 'eval': marker.write_text('{}')
from scripts.mlspace_launch import main
sys.argv = ['launcher', '--phase', sys.argv[2], '--dry']
main()
''', tmp_path, phase)
    assert result.stdout.count(f'"instance_type": "a100.{gpus}gpu"') == count
    assert result.stdout.count(f'scripts.instruct_run {phase}') == count
    assert 'scripts.ar_sps_run' not in result.stdout


def test_instruct_registry_and_signature():
    run_code('''
import experiments as e
from scripts import instruct_config as c
arms = e.collect_experiments()
assert len(arms) == 5
assert [(x.recipe, x.distillation_weight) for x in arms] == [
    ('original_ar',0), ('staged_sps',0), ('staged_sps',.5),
    ('continued_ar',0), ('continued_ar',.5)]
assert c.MODEL == 'Qwen/Qwen3-1.7B' and len(c.REVISION) == 40
assert not arms[0].trainable and e.training_marker(arms[0]) is None
assert c.GLOBAL_BATCH % (4*c.MICRO_BATCH) == 0
for arm in arms[1:]:
    assert arm.num_gpus == 4 and arm.trainable
    command = e.train_command(arm)
    assert 'scripts.instruct_train' in command and '--probe' not in command
    assert '--probe' in e.train_command(arm, probe=True)
    assert e.SUITE in str(e.training_marker(arm))
old = e.design_signature(); c.REVISION = 'changed'
assert old != e.design_signature()
''')


def test_real_instruct_eval_requires_current_preparation(tmp_path, monkeypatch):
    from scripts import mlspace_launch as launch
    monkeypatch.setattr(launch, 'SUITE', SUITE)
    monkeypatch.setattr(launch, 'ARTIFACTS_ROOT', tmp_path)
    monkeypatch.setattr(launch, 'collect_experiments', lambda _: [SimpleNamespace(name='original', trainable=False)])
    monkeypatch.setattr(launch, 'output_dir', lambda _: tmp_path/'original')
    args = SimpleNamespace(phase='eval', dry=False, filter=None)
    with pytest.raises(RuntimeError, match='prepare'):
        launch.submit(args)
    (tmp_path/'prepared.json').write_text(json.dumps(dict(design_signature='old')))
    monkeypatch.setattr(launch, 'design_signature', lambda: 'new')
    with pytest.raises(RuntimeError, match='Stale'):
        launch.submit(args)


def test_missing_training_markers_allow_only_original(tmp_path):
    result = run_code('''
import sys
from pathlib import Path
import experiments as e
e.ARTIFACTS_ROOT=Path(sys.argv[1]); e.RUN_ROOT=e.ARTIFACTS_ROOT/'runs'
from scripts.mlspace_launch import main
sys.argv=['launcher','--dry']
main('eval')
''', tmp_path)
    assert result.stdout.count('"instance_type": "a100.1gpu"') == 1
    assert result.stdout.count('Skip evaluation without trained checkpoint:') == 4
