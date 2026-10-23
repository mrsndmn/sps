# Full SPS MTP continuation with cross entropy

## Hypothesis
Unfreezing the Transformer and tied input/output embeddings improves future-token
predictions beyond embedding-only adaptation of the same trained parent.

## Setup
Training function: `scripts.mtp_train.main`; continuation suite in the
`experiment_configs` registry in `experiments.py`. Evaluation uses
`scripts.mtp_evaluate.evaluate` with a separate original SPS verifier.

## Results
Implementation and correctness/throughput verification are in progress. Parent
adaptation checkpoints and prepared corpus are reused.

## Conclusions
Compare horizon quality and proposal acceptance against the corresponding learned
parent. The unchanged native verifier controls generation correctness; sequential
verification timings do not measure a parallel verifier's potential speedup.

## Changelog
- 2026-10-23: Added full-model continuation with CE-only training from learned parents.
