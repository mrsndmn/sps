# Unified SPS-MTP self-speculation

## Hypothesis
A shared backbone and KV cache can reduce the cost of speculative generation
when the same adapted model proposes and verifies its own predictions.

## Setup
Evaluation function: `scripts.mtp_unified_evaluate.evaluate`. The evaluation-only
registry in `experiments.py` consumes completed full MTP continuation checkpoints.
P1 defines the target distribution; later prediction slots propose after a
pending target anchor. Real states and all prediction groups are computed in a
single layer traversal during verification. Predictions retain their training
mask and positions; they do not see sampled future words.

## Results
Implementation and numerical validation are in progress. No new training is
required. Artifacts use an isolated suite under the existing shared root.

## Conclusions
Compare against native P1 generation of each same adapted checkpoint. This target
can differ from the original SPS checkpoint after continuation. Measure full
latency, acceptance and native agreement; no acceleration is assumed in advance.

## Changelog
- 2026-10-28: Added fused single-backbone self-speculation and same-checkpoint controls.
