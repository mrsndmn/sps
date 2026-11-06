"""Rerun route, checkpoint selection, and resume-stable schedule contracts."""
import json
import math
import os
import subprocess
import sys
from types import SimpleNamespace
import pytest
from scripts import sched_common as sc, sched_config as c


def run(*args):
    return subprocess.run([sys.executable,'-c','import tempfile; from pathlib import Path; from scripts import sched_run as r; t=tempfile.TemporaryDirectory(); r.sc.e.COMMON_ROOT=Path(t.name); r.main()',*args],capture_output=True,text=True,
        env=dict(os.environ,SPS_EXPERIMENT_SUITE=c.SOURCE_SUITE))


@pytest.mark.parametrize('variant,phase,count,gpus', [('step384','eval',20,1),('cosine','train',4,4),('cosine','eval',20,1)])
def test_dry_jobs_are_isolated_and_have_required_gpu_count(variant,phase,count,gpus):
    p=run('--variant',variant,'--phase',phase,'--dry')
    assert p.returncode==0,p.stderr
    assert p.stdout.count(f'"instance_type": "a100.{gpus}gpu"')==count
    assert p.stdout.count('scripts.sched_run --worker')==count
    assert 'qwen3_instruct_original_ar' not in p.stdout
    if phase=='eval':
        for workload in c.WORKLOADS:
            assert p.stdout.count('--workload '+workload)==4


def test_mode_filter_and_invalid_worker_routes():
    p=run('--variant','step384','--phase','eval','--workload','ifeval-nonthinking','--filter','staged_sps_ce','--dry')
    assert p.returncode==0 and p.stdout.count('"instance_type"')==1
    assert run('--variant','step384','--phase','train','--dry').returncode!=0
    assert run('--variant','cosine','--phase','train','--limit','1','--dry').returncode!=0
    assert run('--variant','step384','--phase','eval','--worker','--filter','qwen3_instruct_staged_sps_ce').returncode!=0


def test_legacy_signature_and_old_eval_routes_unchanged():
    p=subprocess.run([sys.executable,'-c','from scripts import sched_common as s; assert s.e.design_signature()==s.c.SOURCE_SIGNATURE; s.signature()'],
        env=dict(os.environ,SPS_EXPERIMENT_SUITE=c.SOURCE_SUITE),capture_output=True,text=True)
    assert p.returncode==0,p.stderr


@pytest.mark.parametrize('total',[102,410,512])
def test_warmup_cosine_endpoints_and_monotonicity(total):
    values=[sc.lr_multiplier(i,total) for i in range(total)]
    warm=math.ceil(.3*total)
    assert values[0]==pytest.approx(1/warm)
    assert values[warm-1]==1 and values[-1]==0
    assert all(a<b for a,b in zip(values[:warm-1],values[1:warm]))
    assert all(a>b for a,b in zip(values[warm-1:-1],values[warm:]))


def test_sps_stage_reset_and_ar_single_schedule():
    from scripts import instruct_config as source
    plan=dict(steps=512,stage_steps=102)
    assert sc.learning_rate('staged_sps',30,plan)==source.EMBED_LR
    assert sc.learning_rate('staged_sps',101,plan)==0
    assert sc.learning_rate('staged_sps',102,plan)==pytest.approx(source.FULL_LR/123)
    assert sc.learning_rate('staged_sps',224,plan)==source.FULL_LR
    assert sc.learning_rate('continued_ar',153,plan)==source.FULL_LR
    # Restoring from a checkpoint never re-starts the warmup.
    for recipe in ('staged_sps','continued_ar'):
        full=[sc.learning_rate(recipe,i,plan) for i in range(512)]
        for cursor in (102,128,384):
            assert full[cursor:]==[sc.learning_rate(recipe,i,plan) for i in range(cursor,512)]


def test_terminal_marker_skips_and_stale_marker_fails(tmp_path,monkeypatch,capsys):
    from scripts import sched_run as r
    exp=SimpleNamespace(name='example',trainable=True)
    monkeypatch.setattr(r,'experiments',lambda _: [exp])
    monkeypatch.setattr(sc,'output_dir',lambda *args:tmp_path)
    monkeypatch.setattr(sc,'signature',lambda:'current')
    monkeypatch.setattr(r.launch,'git',lambda *args:'commit')
    marker=tmp_path/'evaluated.json';marker.write_text(json.dumps({'design_signature':'current'}))
    args=SimpleNamespace(filter=None,phase='eval',variant='step384',workload='ifeval-thinking',dry=True)
    r.submit(args);assert 'Skip completed' in capsys.readouterr().out
    marker.write_text(json.dumps({'design_signature':'old'}))
    with pytest.raises(RuntimeError,match='Stale'):r.submit(args)
