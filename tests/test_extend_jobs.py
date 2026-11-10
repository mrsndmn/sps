"""Epoch exposure, resume ordering and checkpoint-specific job identity."""
import json
import os
import subprocess
import sys
from scripts import extend_config as sc, extend_config as c


def test_each_epoch_is_complete_and_different():
    count,batch,seed=8192,16,1337
    epochs=[]
    for epoch in range(4):
        indices=[i for step in range(epoch*512,(epoch+1)*512)
                 for i in sc.global_batch_indices(step,count,batch,seed)]
        assert sorted(indices)==list(range(count))
        epochs.append(indices)
    assert epochs[0]==list(range(count))
    assert all(epochs[i]!=epochs[i-1] for i in range(1,4))
    for step in (512,700,1024,1536,2047):
        assert sc.global_batch_indices(step,count,batch,seed)==tuple(epochs[step//512][(step%512)*16:(step%512+1)*16])


def test_cli_and_one_gpu_eval_identity(tmp_path):
    env={**os.environ,'SPS_EXPERIMENT_SUITE':c.SOURCE_SUITE}
    # Keep real CLI parsing and payload generation, isolate completion markers.
    cli = '''from pathlib import Path
import sys
from scripts import extend_run as runner
runner.sc.e.COMMON_ROOT = Path(sys.argv.pop(1))
runner.main()
'''
    for phase,step in [('train','2048'),('eval','1024'),('eval','1536'),('eval','2048')]:
        p=subprocess.run([sys.executable,'-c',cli,str(tmp_path),'--phase',phase,'--variant',step,'--dry','--filter','staged_sps_ce'],capture_output=True,text=True,check=True,env=env)
        assert 'scripts.extend_run' in p.stdout
        assert f'step{step}' in p.stdout
        assert ('a100.4gpu' if phase=='train' else 'a100.1gpu') in p.stdout
    code="""from scripts.extend_run import experiments
from scripts import extend_common as sc
exp=experiments('staged_sps_ce')[0]
assert sc.output_dir('1024',exp,'ifeval-thinking')!=sc.output_dir('2048',exp,'ifeval-thinking')
assert sc.output_dir('1024',exp)==sc.output_dir('2048',exp)
"""
    subprocess.run([sys.executable,'-c',code],env=env,check=True)
