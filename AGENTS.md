# Project guidance

Keep experiment definitions and numeric configuration in code. Experiment Markdown
contains only Hypothesis, Setup (training function), Results, Conclusions and Changelog.
Stage files explicitly and commit implementation changes with conventional prefixes.
Do not modify the shared mls library. Use the existing project Python environment.
All evaluation jobs must use exactly one GPU; training/probe allocations remain
separate. Report complete MLSpace UUID job names. Long jobs belong in MLSpace, with durable
Arkhip job_wait observation rather than polling shells.

## Usage arms

- `scripts/train.py`: direct Hydra single-GPU and torchrun DDP training, including
  rank-zero validation, checkpoints, W&B and local `metrics.jsonl` records.
- `scripts/evaluate.py`: one lm-eval task per process, final-checkpoint resolution,
  task-scoped JSON outputs; model behavior comes from the selected experiment.
- `experiments.py`: pilot model registry, data source, paths and train/eval commands.
  `SPS_EXPERIMENT_SUITE=10b` selects the longer comparison. Its artifact root and
  job identities are separate; HF cache, Python headers and committed snapshots
  are shared. The selector propagates into each MLSpace worker and subprocess.
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
  guards; training/eval subprocess logs and terminal markers. Preparation checks
  unique-data coverage for every rank; longer-run estimates also use completed
  pilot runtimes as a conservative floor for fresh throughput probes.
- `scripts/mlspace_launch.py`: dedup, committed-code snapshots, payload creation
  and submission receipts. Jobs consume staged source and the existing shared venv.
- `tests/test_pilot.py` and GitHub Actions: CPU-safe orchestration and configuration
  tests for both suites. `tests/test_long_training.py` checks long-run sampler
  capacity with the installed torch environment. Existing model tests live under
  `src/modeling/tests`; GPU tests require CUDA.

- `SPS_EXPERIMENT_SUITE=mtp`: embedding-only SPS MTP registry in `experiments.py`;
  `run_train_jobs.py`/`run_eval.py` reuse staging/dedup with a separate artifact root.
- `src/modeling/models/sps/mtp.py`: frozen real-state encoder, group-local prediction
  attention, new input embeddings and prefix-only KV cache with rollback.
- `src/modeling/mtp_speculative.py`: generic exact greedy/sampling proposal acceptance,
  residual correction and finite-budget/EOS handling.
- `scripts/mtp_common.py`: source/adaptation loading, matched rank-sharded input
  sequences and document-safe multi-horizon target masks.
- `scripts/mtp_train.py`: torchrun embedding-only adaptation, resume, per-horizon
  validation and lightweight adaptation checkpoints; `scripts/mtp_run.py` wraps
  four-GPU correctness/throughput preparation, bounded training and evaluation.
- `scripts/mtp_evaluate.py`: per-horizon quality and reference-verified greedy and
  sampling generation benchmarks against the original SPS generation path.
- `tests/test_mtp_*.py`: model causality/freeze/cache, probabilistic acceptance,
  benchmark adapters, source registry and launch/data orchestration checks.

- `SPS_EXPERIMENT_SUITE=mtp-unfreeze`: continues each completed embedding-only
  parent with plain CE on all horizons and trains the complete Transformer and
  tied input/output embeddings. Full draft/optimizer checkpoints use an isolated
  artifact root; the sampler starts after the parent's consumed input sequences.
- `scripts/mtp_common.py`: validates parent checkpoints and source/data identities,
  restores full draft weights and preserves tied parameter identity.
- `scripts/mtp_evaluate.py`: updated drafts are verified against a separately loaded
  original SPS; its weights never share parameter storage with the trained draft.
- `tests/test_mtp_unfreeze_*.py`: CE, optimizer parameter deduplication/LR grouping,
  full-checkpoint restoration, warmstart provenance and continuation registry.

- `SPS_EXPERIMENT_SUITE=mtp-parallel`: evaluation-only registry consuming completed
  full-continuation drafts and their learned parents, with isolated artifacts.
- `src/modeling/models/sps/parallel_verify.py`: layer-parallel causal real/predict
  pairs using native q2 attention and verified-prefix KV rollback.
- `scripts/mtp_parallel_evaluate.py`: native numerical audit and matched parallel
  greedy/sampling measurements; records full greedy mismatch counts explicitly.
- `tests/test_mtp_parallel.py`: block causality, EOS, rejection/cache reuse, short
  budgets, CUDA native comparisons and evaluation-only submission routing.

- `SPS_EXPERIMENT_SUITE=spec-baselines`: trainable Medusa-style heads and small AR,
  evaluation-only history lookup and completed MTP controls, with isolated artifacts.
- `src/modeling/spec_baselines.py`: residual heads, document-safe cached causal AR
  and deterministic proposals using only supplied history.
- `scripts/spec_baseline_train.py`/`spec_baseline_run.py`: four-rank CE training,
  exact resume, input provenance, throughput preparation and bounded workers.
- `scripts/spec_baseline_evaluate.py`: target-first anchors, reuse of verifier
  features for heads, conditional AR drafts, full-workload warmup and reached-slot
  acceptance metrics. Existing independent MTP callback behavior stays available.
- `tests/test_spec_baselines.py`: heads freeze/gradient, causal AR cache, proposal
  conditioning, sampling law, history lookup, piggybacked features and dry routing.

- `SPS_EXPERIMENT_SUITE=mtp-unified`: evaluation-only self-speculation from the
  completed full MTP continuations; the adapted P1 branch is its own target.
- `src/modeling/models/sps/unified.py`: one backbone and real-only KV cache;
  fused causal real/group states, all-prefix horizon logits, and rollback.
- `scripts/mtp_unified_evaluate.py`: pending target anchors, P2..PN proposals,
  same-checkpoint native P1 baselines, numerical audits and bounded single-GPU jobs.
- `tests/test_mtp_unified.py`: fused/full-prefix causality, EOS, rejection rollback,
  pass-count invariants, sampling-law oracle and evaluation-only routing.
