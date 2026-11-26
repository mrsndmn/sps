## Hypothesis

Giving length-capped teacher generations more tokens can recover complete reasoning and final answers. Some recovered conversations may exceed the existing training context.

## Setup

Evaluation function is `scripts.teacher_cap_probe.run`.

## Results

Verified 109 archived continuations from ten sources. At a total cap of 2,048 generated tokens, 70 completed and 56 fit the training context. Weighted by the source mix of the 3,196 original capped rejects, these rates were 43.3% and 31.0%. At 1,536 they were 24.1% and 20.0%. Extending to 8,192 raised completion to 72.0% and left trainability at 31.0%. Mean extra tokens per retry were 452, 798 and 3,140 at those caps (1,536, 2,048 and 8,192 respectively).

## Conclusions

Use a total response cap of 2,048 for recovering examples at the existing training context. The recommendation applies to extending saved prefixes on the original prompt cohort. Keep the original prompt eligibility budget when applying this retry policy. The source selector reserves the full initial generation budget inside the training context, so increasing that shared setting would change which prompts enter the corpus. Completion and trainability measure format and length only.

## Changelog

- 2026-11-26: Planned conditional continuation on archived rejected prefixes stratified by SmolTalk source.
- 2026-11-26: Submitted the one-GPU continuation pilot as `lm-mpi-job-8d83b722-4915-4aed-bed1-9441de2f2ece`.
- 2026-11-26: Submitted one-GPU helper jobs for the unfinished continuations. The original job aggregates the complete sample.
  - teacher-cap-assist-0 `lm-mpi-job-7585645e-91a8-4196-abe0-d042a6c251d8`
  - teacher-cap-assist-1 `lm-mpi-job-8f9ca223-22de-481f-8278-7266ff8d45fc`
  - teacher-cap-assist-2 `lm-mpi-job-f71d4bb8-ea98-4bf6-bb2d-289d8d2669fa`
  - teacher-cap-assist-3 `lm-mpi-job-250aa110-6c78-400c-ba5d-9ed4758640ac`
  - teacher-cap-assist-4 `lm-mpi-job-c400f908-6d23-46d9-8835-b49380327911`
  - teacher-cap-assist-5 `lm-mpi-job-f8452a58-516b-422a-9403-51d96f617cef`
  - teacher-cap-assist-6 `lm-mpi-job-9fa08d93-d78a-4605-8aa7-0debef38e4a3`
  - teacher-cap-assist-7 `lm-mpi-job-6da7cb58-4dde-4194-9936-57835ed90713`
- 2026-11-26: Verified all source IDs, archived prefixes, completion checks, training encodings and weighted cap estimates.
