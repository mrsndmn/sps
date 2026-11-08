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
default serialization policy.

Thinking AR and EAGLE-3 completed with actual target replay evidence. Median
decode throughput is 193.55 and 282.44 tokens/s; the paired median EAGLE-3
speedup is 1.46x against AR with graphs. Accuracy remains 62% for both, and
the first-trial token sequences exactly match the eager pilot.

Thinking SPS and SPS-MTP also completed with actual replay evidence. Median
decode throughput is 161.54 and 254.71 tokens/s; the paired median SPS-MTP
speedup is 1.58x (range 1.57–1.58x). Accuracy remains 70% and 63%, and every
first-trial token sequence matches its eager counterpart. Thus the eager
SPS-MTP ratio of 2.04x decreases with graphs; EAGLE-3 has higher absolute
thinking throughput than SPS-MTP despite the lower relative speedup against
its own verifier. Non-thinking graph comparisons remain pending.

## Conclusions

Measure all matched verifier baselines with decode graphs before comparing
their quality and speed with the eager pilot.

## Changelog

- 2026-11-08: Defined isolated CUDA-graph correctness gate and benchmark continuation.
- 2026-11-09: Observed successful capture; replaced callable observer RPC with named worker methods.
- 2026-11-09: Passed actual target replay and greedy parity for both SPS arms; submitted benchmark continuation.
- 2026-11-09: Completed the graph-enabled AR/EAGLE-3 thinking comparison.
- 2026-11-09: Completed SPS thinking with graphs; measured lower relative speedup than the eager pilot.
