---
status: accepted
date: 2026-11-06
---
# Share the SPS-MTP proposal and verification backbone

For the Qwen3 instruction-tuned SPS experiment, MTP proposals and SPS
verification use one backbone. A proposal pass uses a group of prediction
embeddings; a verification pass checks the candidate block using the first
prediction embedding at each causally appropriate prefix. Ordinary SPS decoding
uses the same resulting checkpoint with only that first embedding. This avoids a
second independently trained draft backbone and permits comparison against the
same verifier. The cost is dedicated attention, position and cache handling in
vLLM, and full-backbone MTP adaptation can change the first-slot distribution;
quality must therefore be measured for the resulting model rather than assumed
unchanged from its parent. Each proposal still pays the shared backbone's compute
cost, so this architecture does not guarantee a speedup.

Basis: On 2026-11-06 the user specified a backbone plus multiple prediction
embeddings for proposals and one block forward through the first embedding for
verification, then explicitly confirmed: “Да, единый бэкбон - используется и для
драфта и для проверки”. This followed clarification that the SPS baseline uses
the same final backbone. Existing group-local prediction visibility remains as
accepted in [ADR-0001](0001-group-local-sps-mtp.md); this decision does not replace
that mask policy. Relevant existing contracts are `SPSMTP` in
`src/modeling/models/sps/mtp.py`, `adaptation_loss` in `scripts/mtp_train.py`, and
`PretrainedSPS` in `src/modeling/models/sps/pretrained.py`.
