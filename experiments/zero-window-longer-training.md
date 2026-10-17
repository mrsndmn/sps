# Zero-window SPS longer-training comparison

## Hypothesis
The validation and downstream quality improvement observed in the pilot persists
with a larger training-token budget and a larger set of unique training documents.

## Setup
Training function: `scripts.pilot_run.train`; the longer comparison is defined by
the selected `experiment_configs` registry in `experiments.py`.

## Results
The expanded corpus and both distributed throughput probes are ready. Per-rank
data coverage is sufficient for a full pass through the requested training budget.
Calibrated runtime estimates fit the agreed limit. Both main training jobs have
been submitted; final validation and downstream comparisons remain pending.

## Conclusions
Pending. Compare final quality and training cost against the completed pilot.

## Changelog
- 2026-10-16: Added an isolated longer-training comparison with a shared HF cache.
- 2026-10-18: Verified preparation, data coverage and calibrated runtime estimates;
  submitted both main training jobs.
