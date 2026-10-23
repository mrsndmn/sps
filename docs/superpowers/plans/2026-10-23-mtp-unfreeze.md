# SPS MTP Unfreeze CE Implementation Plan

> **For agentic workers:** Same-session implementation authorized. Use focused independent model and checkpoint/evaluation agents; integrate and invoke portable adversarial-review before jobs.

**Goal:** Continue the existing three trained MTP variants with all Transformer, input embeddings and tied output weights trainable, using CE only.
**Architecture:** Add a separate mtp-unfreeze suite while preserving old frozen behavior by default. Load the existing per-N adaptation vectors plus their original pretrained backbone; create a fresh optimizer. Save complete draft weights and keep a separate unchanged native SPS verifier during evaluation.
**Tech stack:** Existing Python/torch/Triton, torchrun DDP, shared MLSpace staging/dedup/durable waits.
**Spec:** User approved the proposed short adaptation and overrides: unfreeze embedding layer too; parallel N=2,3,4; plain cross entropy; launch now.

## Global Constraints
- Three jobs, each four GPUs; continue from corresponding already trained MTP checkpoint.
- Short additional input budget from source configuration, matched sampler continuation after previous adaptation inputs.
- Plain CE averaged equally across all horizons, including P1. No KL/teacher loss.
- All Transformer and input embeddings train; original tied input/output matrix remains tied and trains once.
- Original SPS-10B verifier remains immutable, even when draft P1 changes.
- Data, environment, source staging and completed runs reused without overwriting old artifacts.
- Source code is single truth for numeric settings. Training runtime measured before full jobs.

## Review Focus
- Real-state no_grad/detach must be disabled for backbone training; gradient through past keys/values required.
- Tied embedding/head parameters must occur in exactly one optimizer group.
- Full draft weights required for correct resume/evaluation; new embeddings alone insufficient.
- Verifier cannot accidentally share updated draft weights or tied embeddings.
- Warmstart and data continuation must match the actual parent checkpoint and sampler position.

### Task 1: Trainable MTP backbone
Files: src/modeling/models/sps/mtp.py; tests/test_mtp_model.py.
Interface: SPSMTP(reference,horizons,backend='auto',train_backbone=False).
- [ ] Test real-stream gradient, tied input/output updates and original frozen behavior.
- [ ] Implement conditional gradient preservation, cache detachment policy and train modes.
- [ ] Verify CPU and CUDA flex backward; commit owned files.

### Task 2: Warmstart/full-checkpoint loading and immutable verifier
Files: scripts/mtp_common.py, scripts/mtp_evaluate.py; tests/test_mtp_unfreeze_loading.py and tests/test_mtp_evaluate.py.
Interfaces: load_mtp(horizons,device,checkpoint_path=None,backend='auto',train_backbone=False), warmstart_checkpoint(horizons), warmstart_metadata(horizons), restore_draft(model,state).
- [ ] Validate parent checkpoint/provenance and restore prelearned embeddings by default in new suite.
- [ ] Restore full checkpoint backbone+vectors and enforce tied parameters.
- [ ] Evaluation before means the trained embedding-only parent; after loads all fine-tuned weights.
- [ ] Native verifier uses separate original SPS reference, never the updated draft.
- [ ] Test provenance, full-state roundtrip and independent verifier; commit owned files.

### Task 3: Registry, training and MLSpace orchestration
Files: experiments.py, scripts/mtp_train.py, scripts/mtp_run.py, scripts/mlspace_launch.py, tests/test_mtp_unfreeze_jobs.py, AGENTS.md, experiment doc and CI.
- [ ] Register separate suite and source parameters, CE all horizons, optimizer groups and full checkpoints.
- [ ] Continue data sequence after parent inputs; verify capacity and warmstart hashes.
- [ ] Prepare tests and actual four-GPU probes, bounded/resumable full runs, evaluation routing.
- [ ] CPU/GPU tests and dry plans; commit and independent adversarial review.
- [ ] Push branch/open PR/check CI; submit prepare job and durable wait.
- [ ] Report measured runtime then launch all three main jobs and observe durably.
