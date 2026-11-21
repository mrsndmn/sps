## Hypothesis

A DFlash drafter trained from scratch on the prepared Qwen3-4B corpus can recover a substantial fraction of the published drafter's acceptance.

## Setup

`scripts.dflash_repro_train.train`

## Results

The distributed GPU probe completed as `lm-mpi-job-8186e5a9-6902-4e42-b249-6a65179aff4b`.
Over the 18 timed steps after warmup, the mean was 2.816 s per step and the
median was 2.779 s. Peak allocated GPU memory was 26.7 GiB per rank at most.
The full training projection is 9.11 hours of wall time, or 36.46 A100 GPU-hours,
before queueing, checkpoint writes and GSM8K evaluation.
Raw measurements are available through `python -m scripts.dflash_repro --phase probe --report`.

Full training completed as `lm-mpi-job-ba89ccd5-484f-40d9-948a-fbfa7215a047`.
MLSpace recorded 9 h 04 min 31 s on four A100s, or 36.30 GPU-hours.
All 11,652 optimizer steps completed. The saved checkpoints at 5,000,
10,000 and 11,652 contain inference weights and matching optimizer and
scheduler states. The training report is available through
`python -m scripts.dflash_repro --phase train --report`.

## Conclusions

The full run took approximately as long as the probe projected.
The checkpoints are ready for acceptance and throughput measurement on GSM8K.

## Changelog

- 2026-11-20 — Added the standalone reproduction and timed distributed probe.
- 2026-11-20 — Recorded probe timings and the full training estimate.
- 2026-11-21 — Verified completed training and all three checkpoints.
