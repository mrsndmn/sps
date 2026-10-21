# Embedding-only MTP adaptation of pretrained SPS

## Hypothesis
A frozen pretrained SPS can support useful multi-token proposals by adapting only
new prediction embeddings, with reference verification preserving generation quality.

## Setup
Training function: `scripts.mtp_train.main`; source registry: `experiment_configs`
in `experiments.py`. Evaluation: `scripts.mtp_evaluate.evaluate`.
Artifacts use a separate suite directory under the existing shared artifact root;
prepared data, source checkpoint, environment and HF cache are reused.

## Results
Implementation and correctness/throughput validation are in progress. No trained
adaptation results or measured generation speedups are available yet.

## Conclusions
Compare per-horizon quality, proposal acceptance and end-to-end generation speed
before deciding whether to unfreeze the transformer. The first verified decoder
uses native sequential SPS verification for reference correctness, so its timing
does not establish the potential speed of a future parallel verifier. Positive
speedup is not an assumed outcome.

## Changelog
- 2026-10-21: Completed interactive requirements interview and began implementation.
