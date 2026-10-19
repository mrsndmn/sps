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
Both main training jobs completed. Final validation NLL is 2.92292 for Standard
and 2.89092 for zero-window SPS, a difference of -0.03199 in favor of SPS.
Training wall time excluding queue was 8.42 and 17.83 hours, respectively.
Both downstream evaluation jobs have been submitted; benchmark results remain
pending. Validation curves and runtime summaries are saved in report artifacts.

## Conclusions
SPS has lower held-out NLL at matched training tokens and uses approximately
twice the training wall time. Downstream results are needed to assess whether
the pilot's task-score improvement persists. Validation uses the expanded corpus's
held-out split, so absolute NLL changes from the smaller pilot are not a strictly
matched-validation comparison; the downstream benchmark sets remain unchanged.

## Changelog
- 2026-10-16: Added an isolated longer-training comparison with a shared HF cache.
- 2026-10-18: Verified preparation, data coverage and calibrated runtime estimates;
  submitted both main training jobs.
- 2026-10-19: Both training runs completed; recorded final validation loss and
  elapsed time, and submitted the downstream evaluations.
