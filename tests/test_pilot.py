import json
from pathlib import Path
import shlex
import subprocess
import sys
import os

import pytest

from scripts import experiment_suite as config
from scripts import mlspace_launch, pilot_run


def test_long_suite_is_tenfold_and_preserves_pilot_paths():
    script = '''
import json
from scripts import experiment_suite as e
from scripts.mlspace_launch import payload
print(json.dumps(dict(root=str(e.ARTIFACTS_ROOT),cache=str(e.HF_HOME),
 names=[x.name for x in e.collect_experiments()],budget=e.TRAIN_TOKENS,
 args=e.overrides(e.collect_experiments()[0]),
 payload=payload('train',e.collect_experiments()[0].name,e.STAGING_ROOT,'a','b'))))
'''
    results = {}
    for suite in ('1b', '10b'):
        env = dict(os.environ, SPS_EXPERIMENT_SUITE=suite)
        results[suite] = json.loads(subprocess.check_output([sys.executable, '-c', script],env=env,text=True))
    short, long = results['1b'], results['10b']
    assert long['budget'] == short['budget'] * 10
    assert Path(long['root']).parent == Path(short['root'])
    assert long['cache'] == short['cache']
    assert set(short['names']).isdisjoint(long['names'])
    assert long['payload']['env_variables']['SPS_EXPERIMENT_SUITE'] == '10b'
    assert long['payload']['job_desc'] != short['payload']['job_desc']
    a, b = (dict(x.split('=',1) for x in r['args']) for r in (short,long))
    assert b['+experiment'] == a['+experiment']
    for key in ('scheduler.warmup_tokens','scheduler.lr_decay_tokens'):
        assert int(b[key]) == int(a[key]) * 10


def test_runtime_forecast_uses_completed_reference_as_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot_run, 'COMMON_ROOT', tmp_path)
    monkeypatch.setattr(pilot_run, 'SANITY_ROOT', tmp_path / 'probe')
    exp = config.ExperimentConfig('new', 's_full_attention_20b', reference_name='old')
    probe = tmp_path / 'probe/out/new'
    probe.mkdir(parents=True)
    (probe/'metrics.jsonl').write_text('\n'.join(json.dumps(dict(event='train',step_seconds=2)) for _ in range(8)))
    old = tmp_path / 'runs/out/old'
    old.mkdir(parents=True)
    (old/'trained.json').write_text(json.dumps(dict(seconds=100)))
    (old/'metrics.jsonl').write_text(json.dumps(dict(event='final',tokens=10*config.TOKENS_PER_STEP)))
    estimate = pilot_run.estimate(exp)
    assert estimate['step_seconds'] == 10
    assert estimate['reference']['name'] == 'old'


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
    assert '\n' not in body['script']


def test_design_signature_tracks_inherited_configs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'REPO_ROOT', tmp_path)
    for name in ('scripts/train.py', 'scripts/evaluate.py', 'conf/model/sps.yaml',
                 'conf/optimizer/adamw.yaml', 'scripts/experiment_suite.py', 'scripts/suite_registry.py',
                 'src/experiments/environment.py', 'src/experiments/pilot.py', 'src/experiments/types.py', 'src/experiments/suite.py'):
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


@pytest.mark.parametrize('pending,failed,completed', [
    ('Pending', 'Failed', 'Completed'), ('pending', 'failed', 'completed'),
    ('PENDING', 'FAILED', 'COMPLETED'),
])
def test_receipt_blocks_duplicate_when_list_api_lags(tmp_path, monkeypatch, pending, failed, completed):
    monkeypatch.setattr(mlspace_launch, 'ARTIFACTS_ROOT', tmp_path)
    receipts = tmp_path / 'submissions'
    receipts.mkdir()
    job = 'lm-mpi-job-48fcb78d-e22e-4076-a28d-f92f4e1d6b0c'
    (receipts / 'receipt.json').write_text(json.dumps(dict(identity='pilot prepare',
        job_name=job, phase='prepare', name='data-and-probes')))
    class Client:
        status = pending
        def get_job_status(self, job_name):
            assert job_name == job
            return dict(status=self.status)
    client = Client()
    active = {}
    mlspace_launch.reconcile_receipts(client, active)
    assert active == {'pilot prepare': job}
    client.status = failed
    active = {}
    mlspace_launch.reconcile_receipts(client, active)
    assert active == {}
    client.status = completed
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


def test_lost_submission_response_blocks_retry_until_reconciled(tmp_path, monkeypatch):
    monkeypatch.setattr(mlspace_launch, 'ARTIFACTS_ROOT', tmp_path)
    body = mlspace_launch.payload('prepare', 'data-and-probes', Path('/stage'), 'a', 'b')
    identity = mlspace_launch.normalize(body['job_desc'])
    class Client:
        calls = 0
        def run_job(self, payload):
            self.calls += 1
            raise TimeoutError('Server accepted job but response was lost')
    client = Client()
    with pytest.raises(TimeoutError):
        mlspace_launch.submit_payload(client, body, 'prepare', 'data-and-probes', 'commit')
    with pytest.raises(RuntimeError, match='Unresolved'):
        mlspace_launch.reconcile_intents({})
    with pytest.raises(RuntimeError, match='Unresolved'):
        mlspace_launch.submit_payload(client, body, 'prepare', 'data-and-probes', 'commit')
    assert client.calls == 1
    job = 'lm-mpi-job-48fcb78d-e22e-4076-a28d-f92f4e1d6b0c'
    mlspace_launch.reconcile_intents({identity: job})
    assert not list((tmp_path / 'submission-intents').glob('*.json'))
    assert json.loads((tmp_path / 'submissions' / f'{job}.json').read_text())['job_name'] == job


def test_submission_transport_never_retries_post():
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread
    from types import SimpleNamespace
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    class Handler(BaseHTTPRequestHandler):
        calls = 0
        def do_POST(self):
            Handler.calls += 1
            self.send_response(503)
            self.end_headers()
        def log_message(self, *args):
            pass
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    session = requests.Session()
    session.trust_env = False
    session.mount('http://', HTTPAdapter(max_retries=Retry(total=3,
        allowed_methods=['POST'], status_forcelist=[503])))
    session.headers['X-Test-Auth'] = 'preserved'
    try:
        mlspace_launch.disable_submission_retries(SimpleNamespace(_session=session))
        response = session.post(f'http://127.0.0.1:{server.server_port}/jobs', timeout=2)
        assert response.status_code == 503
        assert Handler.calls == 1
        assert session.headers['X-Test-Auth'] == 'preserved'
    finally:
        session.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize('suite',['1b','10b','mtp','mtp-unfreeze','mtp-parallel','spec-baselines','mtp-unified'])
def test_all_evaluation_payloads_use_one_gpu_without_changing_training(suite):
    import subprocess,json,os
    code="""
import json
from scripts import experiment_suite as e
from scripts.mlspace_launch import payload
x=e.collect_experiments()[0]
print(json.dumps({phase:payload(phase,x.name,e.STAGING_ROOT,'a','b')['instance_type'] for phase in ['prepare','train','eval']}))
"""
    r=json.loads(subprocess.check_output([sys.executable,'-c',code],env=dict(os.environ,SPS_EXPERIMENT_SUITE=suite),text=True))
    assert r=={'prepare':'a100.4gpu','train':'a100.4gpu','eval':'a100.1gpu'}


def test_staging_uses_linked_worktree_commit(tmp_path, monkeypatch):
    main = tmp_path / "main"
    main.mkdir()

    def git(repo, *args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    git(main, "init", "-q")
    git(main, "config", "user.name", "Test")
    git(main, "config", "user.email", "test@example.com")
    (main / "tracked").write_text("main")
    git(main, "add", "tracked")
    git(main, "commit", "-qm", "test: main")
    worktree = tmp_path / "worktree"
    git(main, "worktree", "add", "-qb", "experiment", str(worktree))
    (worktree / "tracked").write_text("experiment")
    git(worktree, "commit", "-qam", "test: experiment")
    (main / "tracked").write_text("dirty main")

    monkeypatch.setattr(mlspace_launch, "STAGING_ROOT", tmp_path / "stages")
    commit = git(worktree, "rev-parse", "HEAD")
    staged = mlspace_launch.stage_repo(worktree, commit)
    assert git(staged, "rev-parse", "HEAD") == commit
    assert (staged / "tracked").read_text() == "experiment"
    assert (main / "tracked").read_text() == "dirty main"
    assert not (staged / ".git" / "worktrees").exists()


def test_routing_refactor_preserves_known_pilot_design():
    from scripts.design_compatibility import canonical_signature
    migrated = '4780e1816b74ecdf53e3e98bc439c1e5f775274129ebcc5df958a0f56f49169a'
    parent = '344d91c4e7664df9c58cdc44d04fa191886c20f95f3bd271877b1b1e6d899c1f'
    assert canonical_signature(migrated) == parent


def test_parallel_marker_rejects_unfreeze_worker():
    script = '''
from scripts import experiment_suite as e
before = e.design_signature()
e.WORKER_MODULES[e.SUITE] = e.WORKER_MODULES['mtp-unfreeze']
assert e.design_signature() != before
'''
    env = dict(os.environ, SPS_EXPERIMENT_SUITE='mtp-parallel')
    subprocess.run([sys.executable, '-c', script], env=env, check=True)


def test_launcher_rejects_marker_after_worker_change(tmp_path, monkeypatch, capsys):
    marker = tmp_path / 'prepared.json'
    marker.write_text(json.dumps({'design_signature': config.design_signature()}))
    monkeypatch.setattr(mlspace_launch, 'ARTIFACTS_ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['launcher', '--dry', '--phase', 'prepare'])
    assert mlspace_launch.main() == 0
    assert 'Skip completed: data-and-probes' in capsys.readouterr().out
    monkeypatch.setitem(config.WORKER_MODULES, '1b', 'scripts.mtp_run')
    with pytest.raises(RuntimeError, match='Stale completion marker'):
        mlspace_launch.main()
