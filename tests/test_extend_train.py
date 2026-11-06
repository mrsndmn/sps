"""Checkpoint provenance and retained milestones on continuation resume."""
import os
import subprocess
import sys


def test_milestone_precedes_resume_cursor(tmp_path):
    code='''from pathlib import Path
from unittest.mock import patch
from scripts import extend_train as t
root=Path(__import__('sys').argv[1]); paths=[]
def save(path,*args): paths.append(path.name)
with patch.object(t,'save',save):
 t.save_progress(root,None,None,1024,1.,False,{'steps':2048,'stage_steps':102})
 assert paths==['step_1024.pt','latest.pt']
 paths.clear()
 t.save_progress(root,None,None,2048,1.,False,{'steps':2048,'stage_steps':102})
 assert paths==['step_2048.pt','final.pt','latest.pt']
'''
    subprocess.run([sys.executable,'-c',code,str(tmp_path)],check=True,
                   env={**os.environ,'SPS_EXPERIMENT_SUITE':'ar-sps-qwen3-1.7b-instruct'})
