## Hypothesis

Soft Qwen targets can lengthen the accepted prefix of a DFlash drafter relative to fixed token labels on the same dialogues.

## Setup

`scripts.dflash_kd_train.train`

## Results

The eight-A100 probe `lm-mpi-job-296d7c6d-93ef-4d3a-9df1-a33f4ab8b9a1`
matched the hard run's reconstructed starting weight hash. The mean
was 1.819 s over 18 measured steps after warmup. Peak allocated memory
was 28.4 GiB on the highest-memory rank. The full training projection
was 5.89 hours of wall time and 47.1 A100 GPU-hours.

Full training was submitted as `lm-mpi-job-cbb8c358-23dc-4b6d-9b7a-19b9010c13ca`.

## Conclusions

Pending.

## Changelog

- 2026-11-23 — Added a soft-distillation arm from the hard run's initial weights and data schedule.
- 2026-11-23 — Recorded the probe and submitted full training.
