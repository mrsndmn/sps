from pathlib import Path
import json
import numpy as np
import pytest
from workloads.serving import sampling_precision as p


def test_new_allocation_is_one_gpu_and_does_not_reuse_old_roots():
    body=p.payload(p.job_configs[0],Path('/staged'))
    assert body['instance_type']==p.launch.INSTANCE_TYPES_BY_NUM_GPUS[1]
    assert 'scripts/serving/sampling_precision.py --worker' in body['script']
    assert p.ROOT not in (p.audit.ROOT,p.trace.ROOT)
    assert [(x.arm,x.fp32_head) for x in p.experiment_configs]==[
        ('sps_four_slots',False),('sps',True),('sps_four_slots',True),('sps_mtp',True)]


def test_load_matrix_checks_pairing_and_bytes(tmp_path):
    np.savez_compressed(tmp_path/'x.npz',logprobs=np.log([[.5,.5]]))
    meta=dict(cases=['one'],sha256=p.audit.common.file_hash(tmp_path/'x.npz'))
    (tmp_path/'x.json').write_text(json.dumps(meta))
    assert p.load_matrix(tmp_path,'x',[dict(name='one')]).shape==(1,2)
    with pytest.raises(RuntimeError):p.load_matrix(tmp_path,'x',[dict(name='two')])
    meta['sha256']='bad';(tmp_path/'x.json').write_text(json.dumps(meta))
    with pytest.raises(RuntimeError):p.load_matrix(tmp_path,'x',[dict(name='one')])
