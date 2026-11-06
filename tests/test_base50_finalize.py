"""Regression: resuming a last-step latest checkpoint must produce final.pt."""
import os
import subprocess
import sys


def test_final_step_resume_and_interruption(tmp_path):
    code='''import json
from pathlib import Path
from unittest.mock import patch
from scripts import base50_train as t
root=Path(__import__('sys').argv[1])
# Represent an existing latest checkpoint at the final step, without final.pt.
(root/'latest.pt').write_text('last-step state')
plan={'steps':50000}
def crash(*args): raise OSError('interrupted save')
with patch.object(t,'save',crash):
    try: t.finalize(root,None,None,50000,1.,False,plan)
    except OSError: pass
    else: raise AssertionError('save failure swallowed')
assert not (root/'trained.json').exists()
def save(path,model,opt,step,best,stage,plan):
    path.write_text(json.dumps(dict(step=step,stage=stage)))
with patch.object(t,'save',save):
    t.finalize(root,None,None,50000,1.,False,plan)
assert json.loads((root/'final.pt').read_text())['step']==50000
assert Path(json.loads((root/'trained.json').read_text())['checkpoint']).is_file()
'''
    subprocess.run([sys.executable,'-c',code,str(tmp_path)],check=True,
                   env={**os.environ,'SPS_EXPERIMENT_SUITE':'ar-sps-qwen3-1.7b'})
