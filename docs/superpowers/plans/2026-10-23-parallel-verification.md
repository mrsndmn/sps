# Parallel SPS verification implementation plan

**Goal:** Benchmark block verification on existing learned drafts without training.
**Architecture:** Original SPS weights, native prefill, parallel causal real/predict
pairs, and prefix rollback reusing verified real KV. Keep the native generator as
baseline and diagnostic oracle. Separate evaluation artifacts and job identities.
**Tech Stack:** PyTorch, existing Triton paired-query attention, MLSpace.

- [ ] Test block causal conditionals, document boundaries, rejection, full acceptance,
  EOS and short budgets against independent serial reference execution.
- [ ] Implement block verifier with one Transformer layer traversal per draft block,
  retaining verified prefix KV rather than copying/replaying accepted tokens.
- [ ] Measure native numerical drift explicitly; distinguish floating point agreement
  from mathematical speculative acceptance correctness. Fail on greedy mismatches.
- [ ] Add isolated evaluation-only registry/worker, compare learned parents and full
  continuations with the existing greedy/sampling workloads.
- [ ] Verify CPU/GPU and dry launch paths, commit, independent adversarial review,
  push and inspect CI, submit evaluation jobs and register durable observation.
