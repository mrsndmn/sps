import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest


def test_separate_ifeval_dry_plan_preserves_pre_migration_design(tmp_path):
    env = dict(os.environ, SPS_EXPERIMENT_SUITE='ar-sps-qwen3-1.7b-instruct')
    result = subprocess.run([sys.executable, '-c', 'from pathlib import Path; import sys; from workloads.instruction import ifeval_run as r; r.RUN_ROOT=Path(sys.argv[1]); sys.argv=["ifeval","--dry"]; r.main()', str(tmp_path)], env=env,
                            capture_output=True, text=True, check=True)
    assert result.stdout.count('"instance_type": "a100.1gpu"') == 10
    assert result.stdout.count('scripts/instruction/ifeval_run.py --worker --filter') == 10
    assert 'scripts/instruction/instruct_run.py' not in result.stdout


def test_source_gate_rejects_mismatched_preparation(tmp_path, monkeypatch):
    from workloads.instruction import ifeval_run as r
    monkeypatch.setattr(r.e, 'SUITE', r.c.family_config.SOURCE_SUITE)

    monkeypatch.setattr(r.e, 'ARTIFACTS_ROOT', tmp_path)
    (tmp_path/'prepared.json').write_text(json.dumps({'design_signature': 'wrong'}))
    assert r.source_check(SimpleNamespace(trainable=False))['design_signature'] == 'wrong'


def test_ifeval_skips_existing_output(tmp_path, monkeypatch, capsys):
    from workloads.instruction import ifeval_run as r
    exp = SimpleNamespace(name='original', trainable=False)
    monkeypatch.setattr(r, 'RUN_ROOT', tmp_path)
    monkeypatch.setattr(r.e, 'collect_experiments', lambda: [exp])
    monkeypatch.setattr(r, 'signature', lambda: 'current')
    monkeypatch.setattr(r.launch, 'git', lambda *args: 'commit')
    dest = r.output_dir(exp, 'thinking'); dest.mkdir(parents=True)
    marker = dest/'evaluated.json'
    marker.write_text(json.dumps({'design_signature':'current'}))
    args = SimpleNamespace(mode='thinking',filter=None,dry=False,profile="default",author_name='a',telegram_nick='b')
    r.submit(args)
    assert 'Skip existing output' in capsys.readouterr().out
    marker.write_text(json.dumps({'design_signature':'stale'}))
    r.submit(args)
    assert 'Skip existing output' in capsys.readouterr().out
