## Hypothesis

Soft Qwen targets can lengthen the accepted prefix of a DFlash drafter relative to fixed token labels on the same dialogues.

## Setup

`workloads.distillation.dflash_kd_train.train`

## Results

The eight-A100 probe `lm-mpi-job-296d7c6d-93ef-4d3a-9df1-a33f4ab8b9a1`
matched the hard run's reconstructed starting weight hash. The mean
was 1.819 s over 18 measured steps after warmup. Peak allocated memory
was 28.4 GiB on the highest-memory rank. The full training projection
was 5.89 hours of wall time and 47.1 A100 GPU-hours.

Full training completed as `lm-mpi-job-cbb8c358-23dc-4b6d-9b7a-19b9010c13ca`.
MLSpace recorded 5 h 50 min 38 s on eight A100s, or 46.75 GPU-hours.
The checkpoints at steps 5,000, 10,000, and 11,652 were verified.

All three checkpoints were evaluated on the same 1,319 GSM8K prompts as the
hard run. Decode throughput excludes prefill and the first token.

| Checkpoint | Hard tau | Soft tau | Hard acceptance | Soft acceptance | Hard tok/s | Soft tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 3.648 | 3.698 | 17.65% | 17.98% | 74.79 | 76.26 |
| 10,000 | 4.143 | 4.211 | 20.95% | 21.41% | 87.93 | 88.07 |
| 11,652 | 4.161 | 4.224 | 21.07% | 21.49% | 85.34 | 87.95 |

At the final checkpoint, the share of rounds accepting at least one proposal
increased from 83.00% to 83.49%, the share accepting at least eight rose from
9.48% to 9.82%, and the share accepting all fifteen rose from 0.19% to 0.29%.
The final tau gain was 0.063 (paired prompt bootstrap 95% interval
0.051 to 0.075, 10,000 resamples). The interval describes prompt variation
for these checkpoints.

All six trained checkpoints and published b16 emitted identical output tokens
on all 1,319 prompts. On the same 695 prompts that also matched sequential
greedy AR, the final hard and soft checkpoints reached respectively 3.156x
and 3.209x speedup against their same-job AR runs. Published b16 had tau 5.972.
The soft checkpoint reached 70.6% of published b16's throughput on all prompts
in its own job.

- Step 5,000 — `lm-mpi-job-b4740fe3-7b71-4de4-9c42-39390c086ad6`.
- Step 10,000 — `lm-mpi-job-8f423ff1-df20-46cb-ab92-31fbcd65a6ff`.
- Step 11,652 — `lm-mpi-job-418dee86-1b24-4816-9103-6c093b968fa2`.

## Conclusions

Soft distillation improved tau at each evaluated checkpoint. The final gain
was small relative to the gap to published b16. The two full runs used
different GPU counts. The soft run consumed 46.75 GPU-hours and the hard run
consumed 36.30 GPU-hours, so the measured cost difference includes topology
as well as the objective. The comparison uses one training seed.

## Changelog

- 2026-11-23 — Added a soft-distillation arm from the hard run's initial weights and data schedule.
- 2026-11-23 — Recorded the probe and submitted full training.
- 2026-11-23 — Verified all three completed checkpoints and submitted GSM8K evaluation.
- 2026-11-23 — Recorded the matched-step GSM8K comparison against hard targets.
