## Hypothesis

Full decode CUDA graphs can execute the SPS verifier and shared MTP state updates
correctly and change the speedup observed in the eager pilot.

## Setup

Training function: `scripts.vllm_sps_train.main` (reuse its completed checkpoint).
Evaluation function: `scripts.serving_graphs.main` and its `experiment_configs`.

## Results

Pending native graph capture, observed replay and greedy parity checks.

## Conclusions

Measure all matched verifier baselines with decode graphs before comparing
their quality and speed with the eager pilot.

## Changelog

- 2026-11-08: Defined isolated CUDA-graph correctness gate and benchmark continuation.
