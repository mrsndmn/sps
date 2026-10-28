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
All single-GPU evaluations completed. Greedy speedups were 1.45×, 1.56× and
1.58× for the progressively larger horizon groups. Sampling speedups were
1.26×, 1.35× and 1.37×. Each ratio uses native P1 of the same adapted checkpoint.
The fastest configuration produced 140.6 greedy and 123.6 sampled tokens/s,
against native rates of 88.9 and 90.0 tokens/s in its own job.

Every prompt used one prefill and exactly one fused backbone block per
speculative round. Proposal extraction consumed about two percent of total
latency. First additional-proposal sampling acceptance was about 34–35 percent.
Full greedy native equality was 58/64, 60/64 and 61/64 prompts. Numerical audits
had no argmax mismatch on their selected rows, while maximum native probability
drift reached approximately 0.00266.

Detailed counts, acceptance vectors, timing, result hashes and full submission
identifiers are in `run/reports/mtp-unified/results.json` and
`generation_summary.csv`. Source results remain under the shared artifact root.

## Conclusions
The unified implementation demonstrates a measured speedup with shared weights,
shared real KV and a fused verification/prediction pass. The previous expensive
draft-only path is absent. Its target is the adapted model itself, so these ratios
are not a controlled comparison against the earlier original-SPS target setup.

Bitwise native greedy equality is not achieved. Different attention and matrix
shapes produce floating point drift, so exact native generation equivalence must
not be claimed. Repeated timing runs are needed to distinguish close horizon
configurations; these measurements provide no repeatability confidence interval.

## Changelog
- 2026-10-28: Added fused single-backbone self-speculation and same-checkpoint controls.
- 2026-10-28: Completed single-GPU evaluation; recorded speedups, one-pass invariants and native mismatch counts.
