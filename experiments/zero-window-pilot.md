# Zero-window SPS pilot

## Hypothesis
Separating persistent state from prediction improves held-out language modeling
at a matched training-token budget even when prediction tokens cannot read other
prediction tokens.

## Setup
Training function: `scripts.pilot_run.train`; experiment definitions and launch
arguments live in `experiments.py` (`experiment_configs`).

## Results
Both training jobs completed. Final held-out NLL is 3.33603 for Standard and
3.31458 for zero-window SPS, a difference of -0.02145 in favor of SPS. Measured
training wall time was approximately 53 minutes and 111 minutes, respectively,
excluding queue time. Both downstream evaluations completed on all examples,
with no few-shot demonstrations or sample limits.

| Method | ARC-E | HellaSwag | PIQA | SciQ | LAMBADA | Mean |
|--------|------:|----------:|-----:|-----:|--------:|-----:|
| Standard | 40.78 | 28.12 | 57.78 | 59.90 | 18.28 | 40.97 |
| SPS zero-window | 40.87 | 28.21 | 58.71 | 62.50 | 18.20 | 41.70 |

Scores are percentages. The first four tasks use `acc_norm`; LAMBADA uses
`acc`, matching the repository's classic downstream-table metric selection.
The mean is an unweighted average across tasks. Raw metrics, standard errors,
sample counts and source paths are preserved in the report artifacts.

## Conclusions
The shortened single-seed comparison shows lower validation loss and a 0.72
percentage-point increase in mean downstream score at matched training tokens.
SPS improves four tasks, with a small LAMBADA accuracy decrease. It uses
approximately twice the training wall time. This is not a compute-matched
comparison, a statistically established multi-seed improvement, or full-budget
numerical reproduction. A follow-up should repeat seeds before increasing the
training budget, and include a matched-compute baseline if testing efficiency.

## Changelog
- 2026-10-15: Agreed pilot scope and added reproducible MLSpace launch paths.
- 2026-10-15: Prepared reusable tokenized data; corrected pilot sampler coverage
  after the first distributed probe failed.
- 2026-10-16: Completed throughput probes and submitted the paired training jobs.
- 2026-10-16: Both training runs completed; recorded validation loss and elapsed
  time, and submitted downstream evaluations.
- 2026-10-16: Completed all downstream tasks and exported final comparisons,
  raw metrics and validation curves.
