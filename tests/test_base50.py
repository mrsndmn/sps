"""Schedule boundaries and isolated launcher contracts for the long base run."""
import json
import math
import os
import subprocess
import sys
import pytest
from scripts import base50_config as c


def test_schedule_and_resume():
    split=c.STAGE_STEPS
    warmup=math.ceil((c.STEPS-split)*c.WARMUP_FRACTION)
    assert c.learning_rate(split-1)==1e-2
    assert c.learning_rate(split)==pytest.approx(c.FULL_LR/warmup)
    assert c.learning_rate(split+warmup-1)==c.FULL_LR
    assert c.learning_rate(c.STEPS-1)==0
    values=[c.learning_rate(i) for i in range(c.STEPS)]
    assert values[split:split+warmup]==sorted(values[split:split+warmup])
    assert values[split+warmup:]==sorted(values[split+warmup:], reverse=True)
    # Resume derives LR from the absolute optimizer step, without scheduler state.
    for start in (819,1000,15000,32000,49999):
        assert [c.learning_rate(i) for i in range(start,min(start+10,c.STEPS))]==values[start:start+10]
    with pytest.raises(ValueError): c.learning_rate(c.STEPS)


def test_isolation_and_dry_payload():
    env={**os.environ,'SPS_EXPERIMENT_SUITE':c.SOURCE_SUITE}
    code='''import experiments as e
from scripts import base50_config as c
old_root=str(e.ARTIFACTS_ROOT); old_sig=e.design_signature()
e, source=c.configure()
from scripts.base50_run import plan, payload
from pathlib import Path
assert str(e.ARTIFACTS_ROOT)!=old_root
assert e.design_signature()!=old_sig
assert c.configure()[0] is e
assert source.MAX_DATA_ROWS==1600000
assert source.MODEL=="Qwen/Qwen3-1.7B-Base"
assert plan()["steps"]==50000 and plan()["stage_steps"]==819
exps=e.collect_experiments()
assert len(exps)==1 and exps[0].distillation_weight==0
assert payload(exps[0],Path("/tmp/staged"))["instance_type"]==e.INSTANCE_TYPES_BY_NUM_GPUS[4]
'''
    subprocess.run([sys.executable,'-c',code],env=env,check=True)
    result=subprocess.run([sys.executable,'-m','scripts.base50_run','--dry'],env=env,check=True,capture_output=True,text=True)
    data=json.loads(result.stdout)
    assert 'scripts.base50_run --worker' in data['script']
    assert 'qwen3-base-1.7b-ce-50k' in data['job_desc']
