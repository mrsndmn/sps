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
Both downstream evaluations completed with the full zero-shot task sets.
Scores below are percentages: normalized accuracy for ARC-Easy, HellaSwag,
PIQA and SciQ; accuracy for LAMBADA. The mean weights the five tasks equally.

| Model | ARC-Easy | HellaSwag | PIQA | SciQ | LAMBADA | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 48.57 | 33.84 | 63.55 | 70.90 | 28.14 | 49.00 |
| SPS w=0 | 48.53 | 35.02 | 63.93 | 69.00 | 29.87 | 49.27 |

Detailed metrics, sample counts, source paths and complete job identifiers are
saved under the longer suite's report artifacts, alongside validation curves
and runtime summaries.

## Conclusions
SPS has lower held-out NLL at matched training tokens and uses 2.12 times the
training wall time. Its average downstream advantage is 0.27 percentage points,
compared with 0.72 in the pilot. It improves three of five tasks; ARC-Easy and
SciQ normalized accuracy decline. This is a single-seed comparison and does not
establish a statistically reliable advantage. Both models improve their average
benchmark score over the pilot. Validation uses the expanded corpus's held-out
split, so absolute NLL changes from the smaller pilot are not a strictly
matched-validation comparison; the downstream benchmark sets remain unchanged.

## Changelog
- 2026-10-16: Added an isolated longer-training comparison with a shared HF cache.
- 2026-10-18: Verified preparation, data coverage and calibrated runtime estimates;
  submitted both main training jobs.
- 2026-10-19: Both training runs completed; recorded final validation loss and
  elapsed time, and submitted the downstream evaluations.
- 2026-10-19: Verified completed full-set evaluations; recorded benchmark scores,
  metric conventions and the comparison with the pilot.
