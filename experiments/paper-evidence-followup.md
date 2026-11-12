# Hypothesis
The instruction-retention deficit partly reflects response provenance, while the apparent knowledge gain partly reflects fewer invalid completions. Speculative decoding should preserve the final verifier's sampling distribution.

# Setup
`scripts.paper_diagnostics.main` analyzes saved responses. `scripts.sampling_audit.worker` runs the first runtime correctness gate. `scripts.sampling_trace.worker` diagnoses same-prefix distribution differences after a failed gate. `scripts.sampling_precision.worker` tests matched-slot cached decoding and head precision. `scripts.sampling_cache_trace.worker` observes pre-sampler cached target distributions at divergent histories. `scripts.sampling_backbone_precision.worker` tests FP16 backbone execution with the FP32 head retained. `scripts.provenance_train.worker` runs the paired-prompt AR/SPS CE controls. `scripts.provenance_evaluate.evaluate` scores fixed final endpoints on IFEval and both paired held-out target corpora. `scripts.provenance_report.run` validates saved answers and computes paired question contrasts. Executable designs and numerical settings live in those modules.

# Results
Initial paired analysis reproduces the manuscript's shared thinking MMLU subset. The SPS CE gain includes a positive invalid/correct exchange and a negative wrong/correct exchange. Non-thinking IFEval decreases against the source; the CE difference between SPS and continued AR has an interval containing zero. These exploratory intervals describe evaluation questions, not training seeds.

The runtime job completed its checks but failed the greedy-equivalence gate: known-distribution rejection-kernel checks passed; speculative and non-speculative outputs differed in 30 of 112 cases. Eager and graph executions agreed within each arm. One prefix diverged on the first token, before draft-block verification. These are correlated cases over a small prefix set, not independent quality samples.

Same-prefix tracing found exact distribution equality between four-slot non-speculative vLLM and SPS-MTP on all 13 diagnostic histories. Native FP32 one/four-slot maximum total variation was approximately 0.00000853, versus 0.0312 in native BF16. vLLM one/four-slot maximum total variation was approximately 0.0312. This supports a slot-layout-dependent numerical effect, without establishing cached verification equivalence. Initial argmax counts used unstable tie ordering; the derived precision report explicitly records ties and uses first-index argmax.

With matched four-slot execution, cached greedy differences decreased from 16/112 with the BF16 head to 6/112 with the FP32 head. These remaining cases correspond to three distinct prefixes with both EOS modes. The FP32-head one/four-slot full-prefix comparison had zero argmax differences and maximum total variation approximately 0.0102; matched four-slot non-speculative and speculative full-prefix distributions remained identical. Head precision reduces but does not eliminate the cached discrepancy. The diagnostic job completed successfully; this does not pass the original correctness gate.

Cached target-logit observation reproduced all selected generation outputs and validated every observed greedy emitted token against its verifier argmax. On the three first-divergence histories, sequential/speculative total variation was approximately 0.01517, 0.01241 and 0.00826. Cold full-prefix argmax agreed with speculative decoding on all three, but distributions were not identical. Divergences appeared at bonus, second verification and first verification rows, respectively. The discrepancy therefore precedes rejection sampling in these traces; neither exact arithmetic nor cache correctness has yet been established.

The FP16-backbone control with FP32 head reduced matched-slot cached differences to 2/112, both EOS variants of one prefix. The remaining prefix differs from the three affected prefixes in the preceding BF16-backbone/FP32-head control; this is not a nested set of repaired cases. Its first differing token occurred at zero-based output offset 35. Four-slot ordinary/speculative full-prefix distributions were identical on all 13 diagnostic histories. The source export is FP32, so both weight rounding and arithmetic changed. These controls support precision sensitivity but do not eliminate the discrepancy or pass the original BF16 gate.

Paired response preparation retained 5,116 of 5,927 non-thinking training examples and all 2,265 thinking examples; 811 incomplete generated responses were removed from both conditions. Validation retained 83 non-thinking and 33 thinking rows; test retained 88 and 26. Thinking rows are identical, prompt identities match, groups remain split-disjoint, and file hashes and per-mode token accounting validate. Retained non-thinking supervised tokens total 993,455 for corpus versus 1,635,543 for teacher responses. Thus matched prompts do not imply matched token exposure.

All four paired CE controls completed their fixed final endpoint at step 461. On the identical thinking validation targets, final NLL was 0.2643/0.2415 for AR corpus/teacher and 0.2784/0.2584 for SPS corpus/teacher. These are single-run validation measurements, not evidence of instruction-following retention. Non-thinking validation targets differ between conditions, so their NLLs must not be interpreted as directly comparable quality scores.

Completed non-thinking IFEval covers all 541 paired questions. Strict prompt accuracy is 53.79/62.85% for AR corpus/teacher and 50.65/61.37% for SPS corpus/teacher, versus 65.43% for the frozen source. Teacher replacement improves AR by 9.06 pp (paired-question 95% interval 4.99 to 13.12) and SPS by 10.72 pp (6.84 to 14.60). The SPS-minus-AR difference in replacement effects is 1.66 pp (-2.96 to 6.28), so this does not establish a uniquely SPS-specific benefit. SPS teacher remains 4.07 pp below the source (-7.58 to -0.74). These exploratory intervals are unadjusted for multiple comparisons and do not capture training-seed uncertainty.

Crossed held-out non-thinking NLL confirms target-distribution specialization: corpus/teacher-trained AR scores 0.459/1.326 on corpus targets and 0.360/0.135 on teacher targets; SPS scores 0.467/1.287 and 0.366/0.146 respectively. Thus the external instruction-following improvement coexists with worse corpus-target likelihood. Teacher conditions also have lower output-cap rates. Target length, supervised-token exposure and mode weighting remain entangled with response replacement. Thinking benchmark outcomes are pending separately.

# Conclusions
The kernel checks pass, but expanded cached greedy equivalence remains failed. Record this limitation in the manuscript before expanding serving measurements. Further correctness work should separate cache-state/position invariants from numerical effects and test positive-temperature sampling on the actual model; repeated precision variants alone cannot establish equivalence. Paired response-provenance preparation passed integrity checks. The primary AR/SPS CE controls use identical prompts, order and optimizer-step counts, with the final endpoint fixed in advance. Token-normalized loss is retained; response length and token exposure differ, so this estimates the joint effect of replacing targets rather than a length-independent provenance effect. Own-condition validation NLL is monitoring only, not comparable across corpora or used to select the primary endpoint. A token/mode-weight-matched sensitivity study and independent quality evaluation remain necessary before a narrower causal claim. Follow with first-slot-preserving MTP and matched quality/latency/memory evaluation. The initial gate alone cannot prove general sampling equivalence; real-checkpoint block logits, cache rejection paths and independent quality sampling remain necessary.

# Changelog

- 2026-11-12: Recorded completed non-thinking evaluation, paired contrasts and crossed held-out NLL; added reproducible report generation. Thinking observation remains independent.

- 2026-11-12: Submitted one-GPU per-mode evaluation after independent review APPROVE; independent thinking/non-thinking observation registered. continued_ar-corpus--nonthinking `lm-mpi-job-4012a1a0-b56a-4ddd-a6fc-aeaab05c8a47`; continued_ar-corpus--thinking `lm-mpi-job-806ec0f6-8a3f-4df7-81f4-953c7b53ac2f`; continued_ar-teacher--nonthinking `lm-mpi-job-d854372b-6fb0-4e83-aa72-1c20a22aabe5`; continued_ar-teacher--thinking `lm-mpi-job-e2334e27-41e3-4b09-9a2d-f7c669fc24d3`; staged_sps-corpus--nonthinking `lm-mpi-job-66d8885c-ea3a-4bca-98c6-bd479c9def55`; staged_sps-corpus--thinking `lm-mpi-job-b6df9dbe-086a-414b-8821-6e73e81f07d5`; staged_sps-teacher--nonthinking `lm-mpi-job-d887a9e6-f41e-47fd-b26f-ae1faa532350`; staged_sps-teacher--thinking `lm-mpi-job-8bce3211-4a08-4bb5-a47c-b7a52d3eb022`.

- 2026-11-12: Recorded all four completed trainings; added paired IFEval in both modes and crossed held-out NLL at fixed final checkpoints, with submission-pinned checkpoint/protocol identities.

- 2026-11-12: Submitted paired CE controls after independent review APPROVE; durable observation registered. continued_ar-corpus `lm-mpi-job-e6959b4e-039e-4c39-b392-92eac461b145`; continued_ar-teacher `lm-mpi-job-da53e0bd-4a71-457a-9839-f912b6e48918`; staged_sps-corpus `lm-mpi-job-042b5ebe-2a61-40c6-8a99-ab1a17c42e77`; staged_sps-teacher `lm-mpi-job-71bb5f41-846e-479a-887c-5ab27ac0bc05`.

- 2026-11-12: Validated completed paired preparation, measured unequal token exposure, and implemented isolated paired-prompt AR/SPS CE controls with fixed final endpoints.

- 2026-11-11: Recorded completed FP16 control (remaining 2/112 differences) and incorporated the expanded correctness evidence and limitations into the manuscript.

- 2026-11-11: Submitted FP16-backbone control as `lm-mpi-job-769d7189-ca4c-41a6-8e44-1afdaca98630`; focused tests and independent adversarial review passed, durable observation registered.

- 2026-11-11: Recorded completed cached target-logit diagnosis; added an FP16-backbone numerical control. The source export is FP32, so this intervention changes both weight rounding and arithmetic, not arithmetic alone.

- 2026-11-11: Submitted cached target-logit trace as `lm-mpi-job-29d7abea-3ab6-4487-87e2-a6c735cf243c`; focused tests passed and independent adversarial review approved.

- 2026-11-11: Recorded completed precision diagnosis and added target-logit observation at the remaining divergent histories, with output reproduction checks and cold-prefix comparison.

- 2026-11-11: Submitted matched-slot/head diagnostic as `lm-mpi-job-f79d94ac-dc0b-4cfc-aa22-fded96a2e273`; independent review approved and durable observation registered.

- 2026-11-11: Recorded numerical trace evidence; added matched-slot cache and FP32-head diagnosis while preserving failed-gate artifacts.

- 2026-11-11: Submitted distribution diagnosis as `lm-mpi-job-4efd5960-6841-4aff-a889-c8cbefe7632d`; independent review approved and a new durable observation registered.
- 2026-11-11: Recorded a genuine failed greedy gate; added same-prefix full-distribution diagnostics to distinguish slot-count/numerical effects from cached verification effects. The original failed gate remains unchanged.
- 2026-11-10: Submitted runtime audit as `lm-mpi-job-8d5e952d-1f0c-4025-b510-c486b1897e5d`; independent review approved and durable observation registered.
- 2026-11-10: Started saved-answer diagnostics and isolated runtime correctness checks; retained later training and serving stages as dependent follow-ups.
