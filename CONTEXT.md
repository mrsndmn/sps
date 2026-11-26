# SPS Experiments

## Prediction slots and attention

**Zero-window SPS** — An SPS variant in which each prediction slot can attend only to itself and to causally visible input tokens.

**SPS MTP group** — An ordered set of prediction slots that share one visible text prefix and predict consecutive future tokens. Each slot can attend only to that prefix, itself and earlier slots in the same group.

**Prediction horizon** — The distance in tokens from the last visible text token of an SPS MTP group to the token that a slot predicts. The next token has a prediction horizon of one.

## Model adaptation and quality

**AR-to-SPS adaptation** — Conversion of a pretrained autoregressive (AR) language model to zero-window SPS. The adapted model uses dedicated prediction slots for next-token predictions, and the original AR model provides the quality reference.

**Observed quality retention** — A pilot outcome in which every selected benchmark point estimate for the adapted model matches or exceeds the original model’s estimate, and held-out negative log-likelihood stays the same or decreases. These comparisons use matched evaluations and describe only the observed results.

**Pilot comparison** — A shortened experiment that trains SPS and a standard Transformer on the same data with the same training-token budget. It tests the direction of an effect reported in the paper.

## Speculative decoding and verification

**SPS verifier** — The next-token distribution from the first prediction slot of an SPS model at a given prefix. It serves as the reference for accepting or correcting proposed tokens.

**SPS speculative decoding** — A generation method that checks multi-token prediction (MTP) proposals against the SPS verifier. Its acceptance and correction rule preserves the verifier’s output distribution.

**Shared-backbone SPS-MTP** — An SPS model that uses the same backbone weights to propose tokens at multiple prediction horizons and to verify them with the first prediction slot. Ordinary decoding uses the same model with only the first slot.

## Decoding controls and performance

**AR Qwen control** — The original instruction-tuned autoregressive Qwen model that serves as the quality reference and verifies proposals in the EAGLE-3 control.

**EAGLE-3 control** — Speculative decoding that pairs a compatible EAGLE-3 proposer with the AR Qwen control. Its speed baseline is ordinary decoding with the AR Qwen control.

**Matched-verifier speedup** — The ratio of ordinary decoding time to speculative decoding time for the same verifier on a matched workload. Each verifier has its own baseline, and results include absolute latency, throughput and quality.
