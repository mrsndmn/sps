## Hypothesis

A longer draft can increase mean acceptance length and decode throughput even
when the fraction of accepted proposals decreases.

## Setup

Training function `scripts.vllm_sps_train.main`.

## Results

Both drafters accepted slightly more tokens per verification round with the longer
draft. Across methods and modes, mean acceptance length increased by
0.04–0.14 tokens, while decode throughput fell by 18–34%. SPS-MTP had higher
acceptance length and a higher fraction of accepted proposals than EAGLE-3 in
each comparison. The highest measured non-thinking throughput was 335.37
tokens/s for SPS-MTP with the shorter draft. In thinking, the highest was 258.58
tokens/s for EAGLE-3 with the shorter draft.

Each speedup is relative to the same verifier without speculation. Greedy
sequences matched ordinary vLLM SPS on zero of two audited prompts with the
shorter SPS-MTP draft and one of two with the longer draft. All three SPS arms
showed CUDA graph replay. Acceptance counters cover timed requests only.
Complete metrics are in the experiment artifact root under `reports/`.

## Conclusions

The shorter draft provided higher throughput for both methods in both modes.
The additional positions rarely survived verification, and the extra accepted
tokens per round did not offset the cost of drafting and verifying a wider block.
These measurements establish throughput and acceptance for the tested workloads.
The greedy discrepancies rule out a claim of exact output equivalence.

## Changelog

- 2026-11-18: Added the draft-length sweep for EAGLE-3 and shared-backbone SPS-MTP.
- 2026-11-19: Completed the evaluations and recorded throughput, acceptance and greedy diagnostics.
