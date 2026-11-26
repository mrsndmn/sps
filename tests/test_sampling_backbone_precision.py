from pathlib import Path
import sys
from types import SimpleNamespace as NS
import torch
import pytest
from workloads.serving import sampling_backbone_precision as b


def test_one_gpu_isolated_payload():
    result = b.payload(b.job_configs[0], Path('/staged'))
    assert result['instance_type'] == b.launch.INSTANCE_TYPES_BY_NUM_GPUS[1]
    assert 'scripts/serving/sampling_backbone_precision.py --worker' in result['script']
    assert b.ROOT not in (b.precision.ROOT, b.audit.ROOT, b.cache.ROOT)


def test_actual_dtype_and_finite_weights_are_reported():
    extension = b.BackbonePrecisionWorkerExtension()
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float16))
            self.logits_processor = NS(head_dtype=torch.float32)
            self.sps_horizons = 4
    extension.model_runner = NS(model=Model())
    result = extension.sps_backbone_precision_info()
    assert result == dict(parameter_dtypes=['torch.float16'], finite=True, head_dtype='torch.float32', horizons=4)
    with torch.no_grad():
        extension.model_runner.model.weight[0] = float('inf')
    assert not extension.sps_backbone_precision_info()['finite']


@pytest.mark.parametrize('task', ['decode', 'distributions'])
def test_nested_engine_intervention_preserves_head_and_four_slots(tmp_path, monkeypatch, task):
    events = []
    class FakeLLM:
        def __init__(self, **kw):
            events.append(kw)
        def collective_rpc(self, name):
            if name == 'sps_backbone_precision_info':
                return [dict(parameter_dtypes=['torch.float16'], finite=True, head_dtype='torch.float32', horizons=1)]
            if name == 'sps_trace_four_slots':
                self.four = True
                return [dict(horizons=4)]
            if name == 'sps_precision_info':
                return [dict(horizons=4 if getattr(self, 'four', False) else 1, head_dtype='torch.float32')]
        def generate(self):
            return []
    fake = NS(LLM=FakeLLM)
    monkeypatch.setitem(sys.modules, 'vllm', fake)
    monkeypatch.setattr(b, 'root', lambda job: tmp_path)
    b.launch.store_json(tmp_path/'identity.json', dict(audit={}))
    old = tmp_path/'old'
    b.launch.store_json(old/'prefixes.json', [])
    b.launch.store_json(old/'cases.json', [])
    monkeypatch.setattr(b.precision, 'AUDIT_SOURCE', old)
    monkeypatch.setattr(b.precision, 'TRACE_SOURCE', old)
    def decode(*args):
        engine = fake.LLM()
        engine.generate()
    def distributions(*args):
        engine = fake.LLM()
        engine.collective_rpc('sps_trace_four_slots')
        engine.generate()
    monkeypatch.setattr(b.audit, 'decode_worker', decode)
    monkeypatch.setattr(b.trace, 'engine_worker', distributions)
    b.engine_worker(b.experiment_configs[0], task, b.job_configs[0])
    assert events[0]['dtype'] == 'float16'
    assert events[0]['hf_overrides'] == {'head_dtype': 'float32'}
    assert events[0]['worker_extension_cls'].endswith('BackbonePrecisionWorkerExtension')
    assert (tmp_path/'fp16-four'/task/'backbone.json').exists()
    assert fake.LLM is FakeLLM


def test_decode_nonfinite_values_fail_diagnostic():
    with pytest.raises(RuntimeError, match="nonfinite"):
        b.validate_decodes(dict(samples=[dict(tokens=[1], sampled_logprobs=[float("nan")])]))
    b.validate_decodes(dict(samples=[dict(tokens=[1], sampled_logprobs=[-0.2])]))
