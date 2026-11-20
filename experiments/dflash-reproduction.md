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

## Conclusions

The timed probe supports reserving about ten hours on four A100s for training.
The estimate is based on the early training batches and measures training cost.
Drafter acceptance will be measured on the full training checkpoints.

## Changelog

- 2026-11-20 — Added the standalone reproduction and timed distributed probe.
- 2026-11-20 — Recorded probe timings and the full training estimate.
