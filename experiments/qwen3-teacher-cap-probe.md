## Hypothesis

Giving length-capped teacher generations more tokens can recover complete reasoning and final answers. Some recovered conversations may exceed the existing training context.

## Setup

Evaluation function is `scripts.teacher_cap_probe.run`.

## Results

Pending.

## Conclusions

The recommendation applies to extending saved prefixes on the original prompt cohort. Keep the original prompt eligibility budget when applying this retry policy. The source selector reserves the full initial generation budget inside the training context, so increasing that shared setting would change which prompts enter the corpus. Measure marginal token cost and trainability with the existing conversation encoder.

## Changelog

- 2026-11-26: Planned conditional continuation on archived rejected prefixes stratified by SmolTalk source.
- 2026-11-26: Submitted the one-GPU continuation pilot as `lm-mpi-job-8d83b722-4915-4aed-bed1-9441de2f2ece`.
