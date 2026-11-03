# Project guidance

Keep experiment definitions and numeric configuration in code. Experiment Markdown
contains only Hypothesis, Setup (training function), Results, Conclusions and Changelog.
Stage files explicitly and commit implementation changes with conventional prefixes.
Do not modify the shared mls library. Use the existing project Python environment.
All evaluation jobs must use exactly one GPU; training/probe allocations remain
separate. Report complete MLSpace UUID job names. Long jobs belong in MLSpace, with durable
Arkhip job_wait observation rather than polling shells.

## Usage arms

- `scripts/train.py` → direct Hydra/torchrun training; `src/training/sampler.py`, `src/modeling/models/`.
- `scripts/evaluate.py` → direct checkpoint/task evaluation; `experiments.py`.
- `experiments.py` → suite registry, commands and artifact roots; `tests/test_pilot.py`.
- `run_train_jobs.py --phase prepare` → `scripts/mlspace_launch.py` → selected suite preparation worker.
- `run_train_jobs.py --phase train` → `scripts/mlspace_launch.py` → selected suite training worker.
- `run_eval.py` → `scripts/mlspace_launch.py` → selected suite evaluation worker.
- `scripts/pilot_run.py` → pilot/10b workers; `scripts/pilot_data.py`, `tests/test_long_training.py`.
- `scripts/mtp_run.py` → mtp/mtp-unfreeze workers; `scripts/mtp_common.py`.
- `scripts/mtp_train.py` → embedding/full MTP training; `src/modeling/models/sps/mtp.py`.
- `scripts/mtp_evaluate.py` → MTP quality/generation evaluation; `src/modeling/mtp_speculative.py`.
- `scripts/mtp_parallel_evaluate.py` → mtp-parallel worker; `src/modeling/models/sps/parallel_verify.py`.
- `scripts/mtp_unified_evaluate.py` → mtp-unified worker; `src/modeling/models/sps/unified.py`.
- `scripts/spec_baseline_run.py` → spec-baselines workers; `src/modeling/spec_baselines.py`.
- `scripts/spec_baseline_train.py` → speculative-baseline training; `tests/test_spec_baselines.py`.
- `scripts/spec_baseline_evaluate.py` → speculative-baseline evaluation; `tests/test_spec_baselines.py`.
- `scripts/ar_sps_run.py` → ar-sps/ar-sps-kd preparation, training and evaluation workers; `scripts/ar_sps_config.py`.
- `scripts/ar_sps_data.py` → native-tokenizer data and pinned CE data reuse; `tests/test_ar_sps_distillation.py`.
- `scripts/ar_sps_train.py` → CE/KD adaptation; `scripts/ar_sps_common.py`, `src/modeling/models/sps/pretrained.py`.
- `scripts/ar_sps_evaluate.py --audit` → native/SPS numerical preparation gate; `tests/test_ar_sps.py`.
- `scripts/ar_sps_evaluate.py --filter` → checkpoint quality evaluation; `tests/test_ar_sps_jobs.py`.
- `scripts/ar_sps_report.py` → paired baseline/control reports; `tests/test_ar_sps_distillation.py`.
- `.github/workflows/pilot.yml` → CPU orchestration/model tests; `tests/`, `src/modeling/tests/`.

- `SPS_EXPERIMENT_SUITE=ar-sps-{360m,1.7b}-{ce,kd}`: `scripts/ar_sps_config.py` → existing AR/SPS prepare, staged CE training, baseline evaluation and report routes.

- `SPS_EXPERIMENT_SUITE=ar-sps-qwen3-{0.6b,1.7b}` → `scripts/ar_sps_config.py`, shared CE/KD preparation, training and evaluation; `src/modeling/models/sps/pretrained.py`.

- `SPS_EXPERIMENT_SUITE=mmlu-generation run_eval.py` → `scripts/mmlu_run.py`, `scripts/mmlu_generation.py`, `scripts/mmlu_config.py`; `tests/test_mmlu_generation.py`.

- `python -m scripts.report_mmlu_stem` → read-only STEM aggregation; `tests/test_mmlu_stem.py`.

- `SPS_EXPERIMENT_SUITE=mmlu-stem-generation run_eval.py` → `scripts/mmlu_run.py`, `scripts/mmlu_subset.py`; STEM-only evaluation with answer reuse.

- `SPS_EXPERIMENT_SUITE=mmlu-full-resume run_eval.py` → `scripts/mmlu_run.py`, `scripts/mmlu_subset.py`; full-MMLU continuation reusing full/STEM answers.
