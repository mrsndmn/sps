# Full SPS MTP continuation with cross entropy

## Hypothesis
Unfreezing the Transformer and tied input/output embeddings improves future-token
predictions beyond embedding-only adaptation of the same trained parent.

## Setup
Training function: `scripts.mtp_train.main`; continuation suite in the
`experiment_configs` registry in `experiments.py`. Evaluation uses
`scripts.mtp_evaluate.evaluate` with a separate original SPS verifier.

## Results
All continuation and evaluation jobs completed successfully. Each run reused its
learned parent and continued the data stream. Training took 19.19, 22.11 and
24.37 minutes in registry order. Final checkpoints and completed budgets were
verified. Before/after measurements below use matched validation data and prompts
within this evaluation, rather than timings from the earlier parent-only report.

Held-out top-1 accuracy percentages, before to after continuation:

| Model | Horizon 1 | Horizon 2 | Horizon 3 | Horizon 4 |
|---|---:|---:|---:|---:|
| N=2 | 42.14 → 42.01 | 11.57 → 20.22 | — | — |
| N=3 | 42.15 → 41.89 | 11.38 → 19.84 | 7.90 → 11.25 | — |
| N=4 | 42.15 → 41.83 | 11.19 → 19.44 | 7.88 → 11.20 | 6.82 → 8.38 |

Second-horizon NLL fell from 5.90–5.92 to 4.97–5.03; first-horizon NLL increased
from approximately 3.000 to 3.022, 3.030 and 3.035 respectively.

Second-horizon accepted/proposed percentages and end-to-end throughput:

| Model | Greedy acceptance | Sampling acceptance | Greedy tokens/s | Sampling tokens/s | After/native greedy | After/native sampling |
|---|---:|---:|---:|---:|---:|---:|
| N=2 | 18.41 → 55.77 | 21.85 → 33.11 | 33.45 → 36.29 | 33.26 → 34.17 | 0.384 | 0.370 |
| N=3 | 18.39 → 54.06 | 23.45 → 30.43 | 27.66 → 31.26 | 28.35 → 27.50 | 0.336 | 0.301 |
| N=4 | 18.39 → 52.69 | 22.89 → 29.66 | 24.91 → 28.43 | 25.23 → 25.50 | 0.311 | 0.277 |

Fractions include proposals discarded after an earlier rejection, so they are
not conditional acceptance among reached horizons. First-horizon acceptance fell
from approximately 100% to 94.8–96.2% in greedy and 89.3–91.7% in sampling.
Accepted draft tokens per round rose from approximately 1.18–1.20 to 1.52–1.65
in greedy; sampling remained approximately 1.25–1.27 after continuation.

Greedy output matched the original SPS token-for-token for every measured prompt,
before and after continuation. Native-reference logit checks had zero error.
Sampling uses exact proposal acceptance and residual correction against the
original model; equal random seeds do not imply tokenwise equality. Throughput
ratios account for actual emitted lengths, including EOS. Measurements use one
seed and one configured workload, without confidence intervals.

Raw evaluations, quality/count CSVs, timings, preparation and checkpoint provenance
are saved under `run/reports/mtp-unfreeze/` and the suite's shared `report/` directory.

## Conclusions
Full-model CE continuation improves future-token quality and proposal acceptance,
at a small cost to first-horizon quality and alignment with the original verifier.
N=2 has the best measured throughput. Greedy throughput improves by roughly
8–14% over the learned parents, but sampling changes are small and mixed.
No configuration accelerates the original generator: it remains approximately
2.6–3.6 times faster than the adapted speculative path.

The unchanged native verifier controls generation correctness. Its sequential
verification, cache snapshots and accepted-prefix replay are included in timing;
these measurements do not establish a parallel verifier's potential speedup.
Parallel verification is the next implementation question before claiming useful
generation acceleration; further training is not automatically initiated.

## Changelog
- 2026-10-23: Added full-model continuation with CE-only training from learned parents.
- 2026-10-23: Completed all continuations and matched quality/generation evaluations;
  recorded checkpoint provenance, acceptance gains and sequential-verifier timings.
