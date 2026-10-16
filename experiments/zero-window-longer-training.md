# Zero-window SPS longer-training comparison

## Hypothesis
The validation and downstream quality improvement observed in the pilot persists
with a larger training-token budget and a larger set of unique training documents.

## Setup
Training function: `scripts.pilot_run.train`; the longer comparison is defined by
the selected `experiment_configs` registry in `experiments.py`.

## Results
Pending data preparation, calibrated runtime checks, training and evaluation.

## Conclusions
Pending. Compare final quality and training cost against the completed pilot.

## Changelog
- 2026-10-16: Added an isolated longer-training comparison with a shared HF cache.
