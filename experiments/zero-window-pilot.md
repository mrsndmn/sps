# Zero-window SPS pilot

## Hypothesis
Separating persistent state from prediction improves held-out language modeling
at a matched training-token budget even when prediction tokens cannot read other
prediction tokens.

## Setup
Training function: `scripts.pilot_run.train`; experiment definitions and launch
arguments live in `experiments.py` (`experiment_configs`).

## Results
Data preparation and mask checks completed. Throughput probes failed before the
first optimizer step because coarse chunk sharding left a distributed rank empty.
The pilot sampler configuration and shared initial offset have been corrected;
measured runtime estimates, training and evaluation remain pending.

## Conclusions
Pending. A negative or inconclusive difference is a valid pilot outcome.

## Changelog
- 2026-10-15: Agreed pilot scope and added reproducible MLSpace launch paths.
- 2026-10-15: Prepared reusable tokenized data; corrected pilot sampler coverage
  after the first distributed probe failed.
