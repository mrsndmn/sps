"""Dependency-free dry routing for the AR conversion suite."""
import json
import os
import subprocess
import sys
import pytest


@pytest.mark.parametrize('suite', ['ar-sps', 'ar-sps-kd', 'ar-sps-360m-ce', 'ar-sps-1.7b-ce', 'ar-sps-360m-kd', 'ar-sps-1.7b-kd', 'ar-sps-qwen3-0.6b', 'ar-sps-qwen3-1.7b'])
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
        if suite.endswith('-kd') and entry == 'run_eval.py':
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


@pytest.mark.parametrize('suite,model,micro', [
    ('ar-sps-360m-ce', 'HuggingFaceTB/SmolLM2-360M', 2),
    ('ar-sps-1.7b-ce', 'HuggingFaceTB/SmolLM2-1.7B', 1),
])
def test_large_ce_matched_exposure(suite, model, micro):
    code = '''
import sys
from scripts import ar_sps_config as c
from scripts.ar_sps_run import calibrated_plan
import experiments as e
assert c.MODEL == sys.argv[1] and c.MICRO_BATCH == int(sys.argv[2])
assert not c.DISTILLATION and c.LARGE_CE
assert c.GLOBAL_BATCH % (4*c.MICRO_BATCH) == 0
assert all(x.distillation_weight == 0 for x in e.collect_experiments())
assert [(x.recipe, x.trainable) for x in e.collect_experiments()] == [('original_ar', False), ('staged_sps', True)]
assert c.GPU_HOURS == c.PREPARE_HOURS*4+c.TRAIN_HOURS*4+c.EVAL_HOURS*2
p = calibrated_plan({'smollm2_staged_sps': {'embedding': .1, 'full': .2}})
assert p['steps'] == 4096 and p['stage_steps'] == 819 and p['text_tokens'] == 67108864
try:
    calibrated_plan({'smollm2_staged_sps': {'embedding': 100., 'full': 100.}})
except RuntimeError as exc:
    assert 'matched exposure' in str(exc)
else:
    raise AssertionError('Slow run silently reduced token exposure')
assert e.SUITE in str(e.ARTIFACTS_ROOT)
'''
    subprocess.run([sys.executable, '-c', code, model, str(micro)],
                   env=dict(os.environ, SPS_EXPERIMENT_SUITE=suite), check=True)


@pytest.mark.parametrize('suite,reference,model,micro', [
    ('ar-sps-360m-kd', 'ar-sps-360m-ce', 'HuggingFaceTB/SmolLM2-360M', 2),
    ('ar-sps-1.7b-kd', 'ar-sps-1.7b-ce', 'HuggingFaceTB/SmolLM2-1.7B', 1),
])
def test_large_kd_registry(suite, reference, model, micro, tmp_path):
    code = '''
import sys,json
from pathlib import Path
from scripts import ar_sps_config as c
from scripts.ar_sps_run import calibrated_plan
import experiments as e
assert c.DISTILLATION and not c.LARGE_CE
assert (c.REFERENCE_SUITE,c.MODEL,c.MICRO_BATCH)==(sys.argv[1],sys.argv[2],int(sys.argv[3]))
assert c.REFERENCE_CONTROLS == ('smollm2_staged_sps',)
assert c.GPU_HOURS == c.PREPARE_HOURS*4+c.TRAIN_HOURS*4+c.EVAL_HOURS
assert len(e.collect_experiments())==1
exp=e.collect_experiments()[0]
assert exp.trainable and exp.recipe=='staged_sps' and exp.distillation_weight==.5
before=e.design_signature();c.MODEL='different';assert e.design_signature()!=before;c.MODEL=sys.argv[2]
e.COMMON_ROOT=Path(sys.argv[4]);p=e.COMMON_ROOT/c.REFERENCE_SUITE;p.mkdir()
(p/'prepared.json').write_text(json.dumps(dict(design_signature=c.REFERENCE_SIGNATURE,steps=c.MAX_STEPS,stage_steps=int(c.MAX_STEPS*c.STAGE_FRACTION),text_tokens=c.MAX_STEPS*c.GLOBAL_BATCH*c.CONTEXT)))
assert calibrated_plan({'staged':{'embedding':.5,'full':1.5}})['steps']==4096
try:
    calibrated_plan({'staged':{'embedding':100.,'full':100.}})
except RuntimeError as exc:
    assert 'matched exposure' in str(exc)
else:
    raise AssertionError('KD exposure shrank')
'''
    subprocess.run([sys.executable, '-c', code, reference, model, str(micro), str(tmp_path)],
                   env=dict(os.environ, SPS_EXPERIMENT_SUITE=suite), check=True)


@pytest.mark.parametrize('suite', ['ar-sps-qwen3-0.6b', 'ar-sps-qwen3-1.7b'])
def test_qwen_paired_registry(suite):
    code = '''
from scripts import ar_sps_config as c
from scripts.ar_sps_run import calibrated_plan
import experiments as e
assert c.QWEN and not c.DISTILLATION
assert c.MODEL.startswith('Qwen/Qwen3-') and c.MODEL.endswith('-Base')
assert c.MICRO_BATCH==c.EVAL_BATCH==1
arms=e.collect_experiments();assert len(arms)==3
assert not arms[0].trainable and arms[0].name==c.REPORT_BASELINE
assert [(x.recipe,x.distillation_weight) for x in arms[1:]]==[('staged_sps',0),('staged_sps',.5)]
assert c.ENDPOINT_COMPARISONS==((arms[1].name,arms[2].name),)
assert c.GPU_HOURS==4*c.PREPARE_HOURS+8*c.TRAIN_HOURS+3*c.EVAL_HOURS
probes={x.name:dict(embedding=.5,full=1.) for x in arms[1:]}
p=calibrated_plan(probes);assert p['steps']==4096 and p['stage_steps']==819
probes[arms[2].name]['full']=100.
try: calibrated_plan(probes)
except RuntimeError as exc: assert 'matched exposure' in str(exc)
else: raise AssertionError('Slow KD arm reduced exposure')
sig=e.design_signature();c.MODEL='other';assert e.design_signature()!=sig
'''
    subprocess.run([sys.executable,'-c',code],env=dict(os.environ,SPS_EXPERIMENT_SUITE=suite),check=True)


@pytest.mark.parametrize('suite', ['mmlu-generation', 'mmlu-stem-generation', 'mmlu-full-resume'])
def test_mmlu_generation_dry_registry(tmp_path, suite):
    env = dict(os.environ, SPS_EXPERIMENT_SUITE=suite)
    code = """
import sys
from pathlib import Path
import experiments as e
e.COMMON_ROOT = Path(sys.argv[1])
e.ARTIFACTS_ROOT = e.COMMON_ROOT / 'mmlu-generation'
e.RUN_ROOT = e.ARTIFACTS_ROOT / 'runs'
if sys.argv[2] == 'with-markers':
    for exp in e.experiment_configs:
        marker = e.training_marker(exp)
        if marker:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text('{}')
from scripts.mlspace_launch import main
sys.argv = ['launcher', '--dry']
main('eval')
"""
    for mode, count in [('missing', 5), ('with-markers', 14 if suite == 'mmlu-full-resume' else 15)]:
        result = subprocess.run([sys.executable, '-c', code, str(tmp_path), mode], env=env,
                                capture_output=True, text=True, check=True)
        assert result.stdout.count('"instance_type": "a100.1gpu"') == count
        assert result.stdout.count('scripts.mmlu_run eval') == count
        if suite == 'mmlu-full-resume':
            assert 'qwen0.6_ce' not in result.stdout
    rejected = subprocess.run([sys.executable, 'run_train_jobs.py', '--dry'], env=env,
                              capture_output=True, text=True)
    assert rejected.returncode != 0


@pytest.mark.parametrize('tail', [b'{"index":1}', b'{"index":'])
def test_mmlu_resume_uncommitted_final_row(tmp_path, tail):
    from scripts.mmlu_results import recover_samples
    path = tmp_path / 'samples.jsonl'
    first = b'{"index":0}\n'
    path.write_bytes(first + tail)
    assert recover_samples(path) == [{'index': 0}]
    assert path.read_bytes() == first
    with path.open('ab') as stream:
        stream.write(b'{"index":1}\n')
    assert recover_samples(path) == [{'index': 0}, {'index': 1}]


def test_mmlu_resume_rejects_corrupt_committed_rows(tmp_path):
    from scripts.mmlu_results import recover_samples
    path = tmp_path / 'samples.jsonl'
    path.write_bytes(b'{broken}\n{"index":1}\n')
    with pytest.raises(ValueError, match='Corrupt'):
        recover_samples(path)
