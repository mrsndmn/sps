import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

import experiments as config
from scripts import mlspace_launch, pilot_run


def test_pair_has_matched_budget_and_distinct_methods():
    standard, sps = config.collect_experiments()
    assert 'full_attention' in standard.recipe and 'sps_w0' in sps.recipe
    def args(exp):
        return dict(s.split('=', 1) for s in config.overrides(exp) if '=' in s)
    a, b = args(standard), args(sps)
    for key in ('training.max_tokens', 'training.global_batch_size', 'system.data_root',
                'scheduler.warmup_tokens', 'scheduler.lr_decay_tokens'):
        assert a[key] == b[key]
    assert int(a['training.max_tokens']) % config.TOKENS_PER_STEP == 0
    assert all(e.num_gpus == 4 for e in (standard, sps))
    assert len(config.collect_experiments('sps_w0')) == 1


def test_dry_plans_have_no_api_or_filesystem_side_effects(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mlspace_launch, 'ARTIFACTS_ROOT', tmp_path)
    monkeypatch.setattr(mlspace_launch, 'STAGING_ROOT', tmp_path / 'code')
    monkeypatch.setattr(mlspace_launch, 'output_dir', lambda e: tmp_path / e.name)
    monkeypatch.setattr(sys, 'argv', ['launcher', '--dry', '--phase', 'prepare'])
    assert mlspace_launch.main() == 0
    output = capsys.readouterr().out
    payload = json.JSONDecoder().raw_decode(output)[0]
    assert payload['instance_type'] == 'a100.4gpu'
    assert payload['n_workers'] == payload['processes_per_worker'] == 1
    assert not list(tmp_path.iterdir())


def test_eval_invokes_exactly_one_task_per_process():
    exp = config.collect_experiments()[0]
    for task in config.TASKS:
        cmd = config.eval_command(exp, task)
        assert [x for x in cmd if x.startswith('+tasks=')] == ['+tasks=' + task]
        assert '+checkpoint=final' in cmd


def test_estimator_excludes_startup_and_rejects_incomplete_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot_run, 'SANITY_ROOT', tmp_path)
    exp = config.collect_experiments()[0]
    path = tmp_path / 'out' / exp.name / 'metrics.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text('\n'.join(json.dumps(dict(event='train', step_seconds=s))
                              for s in [999, 999] + [2] * 8))
    result = pilot_run.estimate(exp)
    assert result['step_seconds'] == 2
    assert result['token_budget'] == config.TRAIN_TOKENS
    assert result['projected_training_hours'] > 1
    path.write_text(json.dumps(dict(event='train', step_seconds=2)))
    with pytest.raises(RuntimeError, match='eight'):
        pilot_run.estimate(exp)


def test_staging_uses_committed_snapshot(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()
    git('init', '-q')
    git('config', 'user.name', 'Test')
    git('config', 'user.email', 'test@example.com')
    (repo / 'tracked').write_text('original')
    git('add', 'tracked')
    git('commit', '-qm', 'test: snapshot')
    git('remote', 'add', 'origin', 'https://user:credential@example.com/project')
    monkeypatch.setattr(mlspace_launch, 'STAGING_ROOT', tmp_path / 'stages')
    commit = git('rev-parse', 'HEAD')
    staged = mlspace_launch.stage_repo(repo, commit)
    (repo / 'tracked').write_text('changed')
    assert (staged / 'tracked').read_text() == 'original'
    assert 'credential' not in (staged / '.git/config').read_text()
    with pytest.raises(RuntimeError, match='Commit'):
        mlspace_launch.stage_repo(repo, commit)


def test_payload_quotes_staging_path():
    body = mlspace_launch.payload('train', 'sps example', Path('/tmp/path with spaces'), 'a', 'b')
    assert shlex.quote('/tmp/path with spaces') in body['script']
    assert '--filter ' + shlex.quote('sps example') in body['script']


def test_design_signature_tracks_inherited_configs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, '__file__', str(tmp_path / 'experiments.py'))
    for name in ('scripts/train.py', 'scripts/evaluate.py', 'conf/model/sps.yaml',
                 'conf/optimizer/adamw.yaml'):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('initial')
    before = config.design_signature()
    (tmp_path / 'conf/model/sps.yaml').write_text('changed model dimensions')
    assert config.design_signature() != before
    before = config.design_signature()
    (tmp_path / 'conf/optimizer/adamw.yaml').write_text('changed optimizer')
    assert config.design_signature() != before


def test_submission_lock_prevents_concurrent_entry_and_releases(tmp_path, monkeypatch):
    monkeypatch.setattr(mlspace_launch, 'ARTIFACTS_ROOT', tmp_path)
    with mlspace_launch.submission_lock():
        with pytest.raises(RuntimeError, match='Another submission'):
            with mlspace_launch.submission_lock():
                pytest.fail('Concurrent submitter entered critical section')
    assert not (tmp_path / '.submission-lock').exists()
    with pytest.raises(ValueError):
        with mlspace_launch.submission_lock():
            raise ValueError('API failure')
    assert not (tmp_path / '.submission-lock').exists()


def test_receipt_blocks_duplicate_when_list_api_lags(tmp_path, monkeypatch):
    monkeypatch.setattr(mlspace_launch, 'ARTIFACTS_ROOT', tmp_path)
    receipts = tmp_path / 'submissions'
    receipts.mkdir()
    job = 'lm-mpi-job-48fcb78d-e22e-4076-a28d-f92f4e1d6b0c'
    (receipts / 'receipt.json').write_text(json.dumps(dict(identity='pilot prepare',
        job_name=job, phase='prepare', name='data-and-probes')))
    class Client:
        status = 'Pending'
        def get_job_status(self, job_name):
            assert job_name == job
            return dict(status=self.status)
    client = Client()
    active = {}
    mlspace_launch.reconcile_receipts(client, active)
    assert active == {'pilot prepare': job}
    client.status = 'Failed'
    active = {}
    mlspace_launch.reconcile_receipts(client, active)
    assert active == {}
    client.status = 'Completed'
    with pytest.raises(RuntimeError, match='without success marker'):
        mlspace_launch.reconcile_receipts(client, {})


def test_changed_design_cannot_resume_partial_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot_run, 'ARTIFACTS_ROOT', tmp_path)
    monkeypatch.setattr(pilot_run, 'output_dir', lambda e: tmp_path / e.name)
    exp = config.collect_experiments()[0]
    dest = tmp_path / exp.name
    dest.mkdir()
    (tmp_path / 'prepared.json').write_text(json.dumps(dict(
        design_signature=config.design_signature(), estimates={exp.name: dict(
            token_budget=config.TRAIN_TOKENS, projected_training_hours=1)})))
    (dest / 'run-config.json').write_text(json.dumps(dict(design_signature='old')))
    (dest / 'ckpt.pt').touch()
    with pytest.raises(RuntimeError, match='Partial checkpoints'):
        pilot_run.train(exp)
