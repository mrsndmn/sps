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
excluding queue time. Both downstream evaluation jobs have been submitted.

## Conclusions
The shortened single-seed comparison supports a validation-loss improvement at
matched training tokens, with approximately twice the training wall time for SPS.
This is not a compute-matched comparison or full-budget numerical reproduction.
Downstream metrics remain pending.

## Changelog
- 2026-10-15: Agreed pilot scope and added reproducible MLSpace launch paths.
- 2026-10-15: Prepared reusable tokenized data; corrected pilot sampler coverage
  after the first distributed probe failed.
- 2026-10-16: Completed throughput probes and submitted the paired training jobs.
- 2026-10-16: Both training runs completed; recorded validation loss and elapsed
  time, and submitted downstream evaluations.
