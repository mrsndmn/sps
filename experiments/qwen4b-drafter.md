## Hypothesis

Separate mask embeddings may improve the accepted prefix of a parallel draft. The effect may depend on visibility within the draft block.

## Setup

`workloads.distillation.qwen4b_drafter_train.train`

## Results

The four trained drafters reached 25.26–25.75 tokens/s on the full benchmark.
Within each draft length their greedy sequences were identical. Separate
embeddings raised mean accepted length under both masks.

## Conclusions

Separate embeddings have a small observed benefit at this training budget.
Longer drafts rarely extend the accepted prefix, and all four drafters
trail sequential decoding on prompts with matching output.

## Changelog

- 2026-11-20 — Registered the matched mask and embedding comparison.
- 2026-11-20 — Completed the benchmark and added results to the paper appendix.
