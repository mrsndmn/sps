import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest


def test_separate_ifeval_dry_plan_keeps_original_signature(tmp_path):
    env = dict(os.environ, SPS_EXPERIMENT_SUITE='ar-sps-qwen3-1.7b-instruct')
    result = subprocess.run([sys.executable, '-c', 'from pathlib import Path; import sys; from scripts import ifeval_run as r; r.RUN_ROOT=Path(sys.argv[1]); sys.argv=["ifeval","--dry"]; r.main()', str(tmp_path)], env=env,
                            capture_output=True, text=True, check=True)
    assert result.stdout.count('"instance_type": "a100.1gpu"') == 10
    assert result.stdout.count('scripts.ifeval_run --worker --filter') == 10
    assert 'scripts.instruct_run' not in result.stdout
    code = 'import experiments as e; from scripts.ifeval_config import SOURCE_SIGNATURE; assert e.design_signature()==SOURCE_SIGNATURE'
    subprocess.run([sys.executable, '-c', code], env=env, check=True)


def test_source_gate_rejects_mismatched_preparation(tmp_path, monkeypatch):
    from scripts import ifeval_run as r
    monkeypatch.setattr(r.e, 'SUITE', r.c.SOURCE_SUITE)
    monkeypatch.setattr(r.e, 'design_signature', lambda: r.c.SOURCE_SIGNATURE)
    monkeypatch.setattr(r.e, 'ARTIFACTS_ROOT', tmp_path)
    (tmp_path/'prepared.json').write_text(json.dumps({'design_signature': 'wrong'}))
    with pytest.raises(RuntimeError, match='preparation'):
        r.source_check(SimpleNamespace(trainable=False))


def test_completed_ifeval_is_skipped_and_stale_marker_rejected(tmp_path, monkeypatch, capsys):
    from scripts import ifeval_run as r
    exp = SimpleNamespace(name='original', trainable=False)
    monkeypatch.setattr(r, 'RUN_ROOT', tmp_path)
    monkeypatch.setattr(r.e, 'collect_experiments', lambda _: [exp])
    monkeypatch.setattr(r, 'signature', lambda: 'current')
    monkeypatch.setattr(r.launch, 'git', lambda *args: 'commit')
    dest = r.output_dir(exp, 'thinking'); dest.mkdir(parents=True)
    marker = dest/'evaluated.json'
    marker.write_text(json.dumps({'design_signature':'current'}))
    args = SimpleNamespace(mode='thinking',filter=None,dry=True,author_name='a',telegram_nick='b')
    r.submit(args)
    assert 'Skip completed: original' in capsys.readouterr().out
    marker.write_text(json.dumps({'design_signature':'stale'}))
    with pytest.raises(RuntimeError,match='Stale'):
        r.submit(args)
