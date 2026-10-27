# Speculative baseline comparison plan

Goal: compare prompt lookup, lightweight Medusa-style heads and a small causal AR
model with completed SPS-MTP drafts using the same original SPS verifier.

- Define isolated experiments and numeric budgets in experiments.py; use existing
  FineWeb memmaps, environment, source checkpoint and four-GPU MLSpace launchers.
- Implement heads predicting future tokens from frozen original P1 features, with
  training only in the heads. During generation reuse verifier features, with a
  target-generated anchor and no additional full SPS proposal pass.
- Implement small causal AR training/cache and proposals conditioned on the actual
  preceding sampled draft tokens. Extend the acceptance callback protocol without
  changing existing independent-MTP behavior.
- Implement deterministic history-only lookup, including no-match fallback.
- Record reached-position and accepted-position counts as well as unconditional
  proposal counts. Compare all methods with a common target-first anchor policy.
- Keep evaluation warmups and verifier kernels matched across methods, warm the
  complete greedy/sampling workload before timing to cover changing block shapes.
- Test causality, gradients/freeze, sampler/resume, proposal probabilities, EOS,
  rejection, cache feature alignment, provenance and isolated dry submission.
- Commit, independent adversarial review, CI, then four-GPU data/throughput
  preparation; launch training only when estimates fit the configured cap.
