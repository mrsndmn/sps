# Project guidance

Keep experiment definitions and numeric configuration in code. Experiment Markdown
contains only Hypothesis, Setup (training function), Results, Conclusions and Changelog.
Stage files explicitly and commit implementation changes with conventional prefixes.
Do not modify the shared mls library. Use the existing project Python environment.
Report complete MLSpace UUID job names. Long jobs belong in MLSpace, with durable
Arkhip job_wait observation rather than polling shells.

## Usage arms

- `scripts/train.py`: direct Hydra single-GPU and torchrun DDP training, including
  rank-zero validation, checkpoints, W&B and local `metrics.jsonl` records.
- `scripts/evaluate.py`: one lm-eval task per process, final-checkpoint resolution,
  task-scoped JSON outputs; model behavior comes from the selected experiment.
- `experiments.py`: pilot model registry, data source, paths and train/eval commands.
- `src/training/sampler.py`: shared rank-independent initial offset, distributed
  sequence/chunk sharding and exact per-rank resume order; regression tests in
  `src/modeling/tests/core/test_sampler.py` cover the small pilot corpus.
- `run_train_jobs.py --phase prepare`: MLSpace data preparation and both four-GPU
  throughput probes. `--phase train` submits the two paired training jobs only
  after successful preparation. `--dry` plans without API calls or writes.
- `run_eval.py`: evaluation submission for completed training only; `--dry` skips
  absent checkpoints successfully. Re-running skips successful markers and active jobs.
- `scripts/pilot_data.py`: pinned FineWeb-Edu shard reuse/download, hash-based
  document split, GPT-2 encoding and uint16 memmaps with completion manifest.
- `scripts/pilot_run.py`: worker-side preparation, throughput estimates and timeout
  guards; training/eval subprocess logs and terminal markers.
- `scripts/mlspace_launch.py`: dedup, committed-code snapshots, payload creation
  and submission receipts. Jobs consume staged source and the existing shared venv.
- `tests/test_pilot.py` and GitHub Actions: CPU-safe orchestration and configuration
  tests. Existing model tests live under `src/modeling/tests`; GPU tests require CUDA.
