# Parallel SPS verification

## Hypothesis
Verifying proposed token blocks in a single layer traversal and retaining accepted
prefix KV reduces speculative decoding overhead on existing learned MTP drafts.

## Setup
Evaluation function: `scripts.mtp_parallel_evaluate.evaluate`. The evaluation-only
registry in `experiments.py` selects completed drafts and their learned parents.
The original SPS generator supplies baseline timings and a numerical oracle.

## Results
Implementation and independent verification are in progress. Numerical agreement,
greedy sequence equality and throughput are reported separately. No training is
required for this experiment.

## Conclusions
Block GEMM shapes may differ numerically from native single-token execution.
Speculative acceptance preserves the computed verifier probabilities; bitwise
identity with native sampling is not assumed. Generation acceleration must be
established by complete-workload timings, including proposal and cache recovery.

## Changelog
- 2026-10-23: Added parallel block verification for completed MTP drafts.
