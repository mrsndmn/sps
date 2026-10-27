# Additional speculative proposal baselines

## Hypothesis
History lookup, lightweight target-feature heads and a smaller autoregressive
model can reveal whether SPS-MTP's main limitation is proposal cost, proposal
quality or the absence of sampled-token conditioning within its draft block.

## Setup
Training function: `scripts.spec_baseline_train.main`; evaluation function:
`scripts.spec_baseline_evaluate.evaluate`. The `experiment_configs` registry in
`experiments.py` defines training budgets, models, evaluation horizons and workers.
Existing SPS-MTP continuations are controls in the same evaluation suite.

The heads are a Medusa-style chain adaptation, not a reproduction of Medusa tree
search. They predict later positions from the original target's previous-prefix
features, reused from verification without an extra full Transformer proposal
pass. Every method starts with an original-target anchor. Conditional AR proposal
probabilities are evaluated on the actual earlier sampled draft tokens.

## Results
Implementation and four-GPU throughput preparation are in progress. Existing
FineWeb memmaps and the original SPS checkpoint are reused. No new download is
needed. The heads inherit a pretrained frozen encoder; the small AR model starts
from scratch, which is an explicit difference in their initialization budgets.

## Conclusions
Compare complete-workload throughput, emitted tokens per verification round,
reached-position conditional acceptance, unconditional proposal survival and
native greedy agreement. Block floating point differences remain a limitation;
no exact native-output equivalence is assumed. Full evaluation workloads are
warmed before timing to cover the tested block-size paths.

## Changelog
- 2026-10-27: Added history lookup, target-feature heads, small causal AR and
  rerun controls with matched anchors, counters and warmup policy.
