## Hypothesis

Full decode CUDA graphs can execute the SPS verifier and shared MTP state updates
correctly and change the speedup observed in the eager pilot.

## Setup

Training function: `scripts.vllm_sps_train.main` (reuse its completed checkpoint).
Evaluation function: `scripts.serving_graphs.main` and its `experiment_configs`.

## Results

Native target graph capture and replay passed for SPS and SPS-MTP. Both arms
exactly matched the training-runtime greedy reference on the correctness pilot.
The observer recorded 30 target replays for SPS and 12 for SPS-MTP; capture alone
is not accepted as replay evidence. Named worker-extension RPC methods keep the
default serialization policy. Quality and speed measurements remain pending.

## Conclusions

Measure all matched verifier baselines with decode graphs before comparing
their quality and speed with the eager pilot.

## Changelog

- 2026-11-08: Defined isolated CUDA-graph correctness gate and benchmark continuation.
- 2026-11-09: Observed successful capture; replaced callable observer RPC with named worker methods.
- 2026-11-09: Passed actual target replay and greedy parity for both SPS arms; submitted benchmark continuation.
