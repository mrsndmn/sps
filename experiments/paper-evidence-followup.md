# Hypothesis
The instruction-retention deficit partly reflects response provenance, while the apparent knowledge gain partly reflects fewer invalid completions. Speculative decoding should preserve the final verifier's sampling distribution.

# Setup
`scripts.paper_diagnostics.main` analyzes saved responses. `scripts.sampling_audit.worker` runs the first runtime correctness gate. `scripts.sampling_trace.worker` diagnoses same-prefix distribution differences after a failed gate. Executable designs and numerical settings live in those modules.

# Results
Initial paired analysis reproduces the manuscript's shared thinking MMLU subset. The SPS CE gain includes a positive invalid/correct exchange and a negative wrong/correct exchange. Non-thinking IFEval decreases against the source; the CE difference between SPS and continued AR has an interval containing zero. These exploratory intervals describe evaluation questions, not training seeds.

The runtime job completed its checks but failed the greedy-equivalence gate: known-distribution rejection-kernel checks passed; speculative and non-speculative outputs differed in 30 of 112 cases. Eager and graph executions agreed within each arm. One prefix diverged on the first token, before draft-block verification. These are correlated cases over a small prefix set, not independent quality samples.

# Conclusions
Run the kernel/cache gate before expanding serving measurements. Next implement matched response-provenance preparation and AR/SPS CE controls, then select a recipe using separate validation data before broader evaluation and seed replication. Follow with first-slot-preserving MTP and matched quality/latency/memory evaluation. The initial gate alone cannot prove general sampling equivalence; real-checkpoint block logits, cache rejection paths and independent quality sampling remain necessary.

# Changelog
- 2026-11-11: Recorded a genuine failed greedy gate; added same-prefix full-distribution diagnostics to distinguish slot-count/numerical effects from cached verification effects. The original failed gate remains unchanged.
- 2026-11-10: Submitted runtime audit as `lm-mpi-job-8d5e952d-1f0c-4025-b510-c486b1897e5d`; independent review approved and durable observation registered.
- 2026-11-10: Started saved-answer diagnostics and isolated runtime correctness checks; retained later training and serving stages as dependent follow-ups.
