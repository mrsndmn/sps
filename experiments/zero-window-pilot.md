# Zero-window SPS pilot

## Hypothesis
Separating persistent state from prediction improves held-out language modeling
at a matched training-token budget even when prediction tokens cannot read other
prediction tokens.

## Setup
Training function: `scripts.pilot_run.train`; experiment definitions and launch
arguments live in `experiments.py` (`experiment_configs`).

## Results
Data preparation, mask checks and both distributed throughput probes completed.
Measured runtime forecasts fit the agreed training limit. Both main training jobs
have been submitted; final quality comparisons and downstream evaluation remain
pending. Runtime measurements and forecasts are saved in the preparation artifact.

## Conclusions
Pending. A negative or inconclusive difference is a valid pilot outcome.

## Changelog
- 2026-10-15: Agreed pilot scope and added reproducible MLSpace launch paths.
- 2026-10-15: Prepared reusable tokenized data; corrected pilot sampler coverage
  after the first distributed probe failed.
- 2026-10-16: Completed throughput probes and submitted the paired training jobs.
