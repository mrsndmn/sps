# Parallel SPS verification

## Hypothesis
Verifying proposed token blocks in a single layer traversal and retaining accepted
prefix KV reduces speculative decoding overhead on existing learned MTP drafts.

## Setup
Evaluation function: `scripts.mtp_parallel_evaluate.evaluate`. The evaluation-only
registry in `experiments.py` selects completed drafts and their learned parents.
The original SPS generator supplies baseline timings and a numerical oracle.

## Results
All parallel evaluation jobs completed. Implementation passed the focused CPU/CUDA
test suite, independent adversarial review and CI. Existing learned parents and
full-continuation checkpoints were evaluated without additional training.

End-to-end throughput after full continuation, with the native generator measured
in each job as baseline:

| Model | Greedy tokens/s | Greedy/native | Sampling tokens/s | Sampling/native |
|---|---:|---:|---:|---:|
| N=2 | 85.70 | 0.863 | 75.95 | 0.770 |
| N=3 | 95.01 | 0.953 | 80.78 | 0.829 |
| N=4 | 86.90 | 0.941 | 74.49 | 0.795 |

Corresponding learned parents achieved greedy/native ratios of 0.581, 0.594 and
0.603, and sampling/native ratios of 0.772, 0.829 and 0.806. Full continuation
substantially helps greedy acceptance but does not improve sampling throughput
in these measurements. After continuation, second-horizon acceptance was
53.1–55.7% in greedy and 29.6–32.7% in sampling. Accepted draft tokens per round
were 1.51–1.65 and 1.24–1.26 respectively.

Relative to the earlier sequential-verifier report, normalized throughput ratios
improved approximately 2.1–3.0 times. That comparison uses separate jobs; absolute
native baseline throughput also varied, so it is not a controlled same-job
comparison between verifier implementations. Actual baseline throughput ranged
from 92.4 to 99.7 tokens/s across the current workloads.

Every before/after run matched the complete native greedy output on 62 of 64
prefixes. Thus none meets the prior bitwise native greedy-equivalence criterion.
The held-out numerical audits had no argmax mismatches, but maximum probability
errors reached approximately 0.000437 and logit errors 0.00904. Those audits did
not cover every generated prefix.

Independent replay of both first divergences for N=2 reproduced the recorded
native and parallel argmax choices. The native top-two probability margins were
0.000175 and 0.000219, below observed maximum probability differences of 0.000495
and 0.000606 at those prefixes. Small numerical differences can therefore change
argmax and the subsequent continuation. These replay diagnostics establish the
cause for the two inspected N=2 cases, not every mismatch in the other runs.

Raw evaluations, native/parallel mismatches, independent replay diagnostics and
summary rates are saved in `run/reports/mtp-parallel/` and the suite's shared
`report/` directory. Report generation checked completion identities, raw
acceptance counts, throughput ratios and all sequence-equality counts.

## Conclusions
Parallel verification removes much of the prior overhead. The best measured
configuration is the full-continuation draft with three horizons, reaching 0.953
of native greedy throughput. None accelerates the original SPS generator in this
workload. Timings include proposal, prefill, verification, correction/bonus fusion,
cache reconciliation and acceptance logic.

Block GEMM shapes differ numerically from native single-token execution, and full
greedy agreement fails on a small subset of prompts. Speculative acceptance and
residual correction preserve the computed verifier probabilities; bitwise identity
with native sampling is not established. These timings must not be presented as
acceleration with exact native output preservation. Numerical alignment and further
profiling are separate next steps; no additional training is initiated.

## Changelog
- 2026-10-23: Added parallel block verification for completed MTP drafts.
- 2026-10-23: Completed all evaluations; recorded throughput, native output
  mismatches and independent replay of the first N=2 divergences.
