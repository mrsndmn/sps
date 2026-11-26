import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from workloads.serving import sampling_cache_trace as c


def test_row_histories_follows_metadata_not_physical_order():
    assert c.row_histories([7, 8], [3, 4], [1, 0], [2], 3) == [
        (1, [7, 8]), (0, [7, 8, 3]), (2, [7, 8, 3, 4])]
    with pytest.raises(RuntimeError):
        c.row_histories([7], [3], [0], [0], 2)


def make_observer(tmp_path, emitted):
    runner = NS(use_async_scheduling=False,
                model=NS(vllm_config=NS(model_config=NS(enforce_eager=True)),
                         logits_processor=NS(head_dtype=torch.float32), sps_horizons=4),
                input_batch=NS(req_ids=['one'], num_tokens_no_spec=[2],
                               token_ids_cpu=np.array([[7, 8, 3, 4]])))
    logits = torch.tensor([[0., 0., 0., 0., 4.], [0., 0., 0., 4., 0.], [4., 0., 0., 0., 0.]])
    before = logits.clone()
    def sample(values, metadata):
        assert torch.equal(values, before)
        values.zero_()  # Observer must preserve pre-sampler probabilities.
        return NS(sampled_token_ids=torch.tensor([emitted]))
    runner._sample = sample
    observer = c.CacheTraceWorkerExtension()
    observer.model_runner = runner
    cases = [dict(name='target', tokens=[7, 8, 3])]
    observer.sps_cache_trace_start(cases, str(tmp_path))
    metadata = NS(num_draft_tokens=[2], draft_token_ids=torch.tensor([3, 4]),
                  target_logits_indices=torch.tensor([1, 0]), bonus_logits_indices=torch.tensor([2]))
    return observer, runner, cases, logits, metadata, before


def test_observer_captures_pre_sampler_logits_and_validates_all_accepted(tmp_path):
    observer, runner, cases, logits, metadata, before = make_observer(tmp_path, [3, 4, 0])
    runner._sample(logits, metadata)
    observer.sps_cache_trace_finish()
    matrix = c.precision.load_matrix(tmp_path, 'logprobs', cases)
    np.testing.assert_allclose(matrix[0], torch.log_softmax(before[0], -1).numpy())
    records = json.loads((tmp_path / 'steps.json').read_text())
    assert records[0]['emitted'] == [3, 4, 0]
    assert records[0]['rows'][1]['case'] == 'target'


def test_observer_accepts_rejection_padding_and_detects_wrong_sampler(tmp_path):
    observer, runner, _, logits, metadata, _ = make_observer(tmp_path, [3, -1, -1])
    runner._sample(logits, metadata)
    observer.sps_cache_trace_finish()
    observer, runner, _, logits, metadata, _ = make_observer(tmp_path, [2, -1, -1])
    with pytest.raises(RuntimeError, match='disagrees'):
        runner._sample(logits, metadata)


def test_observer_rejects_missing_history_and_async(tmp_path):
    observer, runner, _, _, _, _ = make_observer(tmp_path, [3, -1, -1])
    with pytest.raises(RuntimeError, match='not reproduced'):
        observer.sps_cache_trace_finish()
    runner.use_async_scheduling = True
    with pytest.raises(RuntimeError, match='synchronous'):
        observer.sps_cache_trace_start([], str(tmp_path))


def test_allocation_is_one_gpu_and_outputs_are_isolated():
    body = c.payload(c.job_configs[0], Path('/staged'))
    assert body['instance_type'] == c.launch.INSTANCE_TYPES_BY_NUM_GPUS[1]
    assert 'scripts/serving/sampling_cache_trace.py --worker' in body['script']
    assert c.ROOT not in (c.precision.ROOT, c.audit.ROOT, c.trace.ROOT)
