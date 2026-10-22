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
Preparation completed with 62 correctness tests passing on the worker. Native
reference logits and greedy outputs matched exactly on the source-checkpoint
checks. Measured throughput was approximately 121.7k, 87.3k and 69.5k input tokens/s
for the three horizon counts in registry order. Conservative training-time ranges
are 3.0–4.5, 4.0–5.8 and 4.9–7.0 hours, excluding queue time. All adaptation jobs
have been submitted. Final quality and generation-speed results remain pending.

## Conclusions
Compare per-horizon quality, proposal acceptance and end-to-end generation speed
before deciding whether to unfreeze the transformer. The first verified decoder
uses native sequential SPS verification for reference correctness, so its timing
does not establish the potential speed of a future parallel verifier. Positive
speedup is not an assumed outcome.

## Changelog
- 2026-10-21: Completed interactive requirements interview and began implementation.
- 2026-10-22: Verified preparation, reported measured runtime estimates and submitted
  the three adaptation jobs with durable observation.
