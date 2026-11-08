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
its own verifier.

Non-thinking AR and EAGLE-3 achieve 199.48 and 319.62 tokens/s, with a paired
median EAGLE-3 speedup of 1.60x and unchanged accuracy of 57% and 63%.
Non-thinking SPS and SPS-MTP achieve 185.60 and 334.87 tokens/s, with a paired
median SPS-MTP speedup of 1.80x (range 1.80–1.81x) and unchanged accuracy of
55% and 59%. Median request latency is 0.301 and 0.168 seconds for this SPS pair.

All evaluation arms passed current source, runtime, checkpoint and input
consistency checks and recorded actual target replay. Every token sequence in
every timing repeat matches the corresponding eager run. The generated artifact
reports are `reports/summary.csv`, `reports/results.json` and
`reports/eager-vs-graphs.csv` under the graph experiment root.

## Conclusions

Decode graphs substantially increase absolute throughput for all four arms and
reduce the relative speculative gains compared with eager execution. SPS-MTP
speedups decrease from 2.04x/2.32x to 1.58x/1.80x in thinking/non-thinking;
EAGLE-3 decreases from 1.59x/1.75x to 1.46x/1.60x. Ratios use separate matched
verifier baselines. EAGLE-3 has higher absolute thinking throughput; SPS-MTP has
higher absolute non-thinking throughput on this pilot.

The measured scope is full target decode graphs without Torch compilation;
prefill remains outside the graph. Replay evidence verifies the target rather
than the separate EAGLE drafter. Quality limitations from the eager pilot remain:
the small sampled benchmark and its output caps do not establish unchanged
general quality, and SPS-MTP thinking accuracy is lower than SPS accuracy.

## Changelog

- 2026-11-08: Defined isolated CUDA-graph correctness gate and benchmark continuation.
- 2026-11-09: Observed successful capture; replaced callable observer RPC with named worker methods.
- 2026-11-09: Passed actual target replay and greedy parity for both SPS arms; submitted benchmark continuation.
- 2026-11-09: Completed the graph-enabled AR/EAGLE-3 thinking comparison.
- 2026-11-09: Completed SPS thinking with graphs; measured lower relative speedup than the eager pilot.
- 2026-11-09: Completed all non-thinking arms, verified exact eager token parity, and generated final comparison reports.
