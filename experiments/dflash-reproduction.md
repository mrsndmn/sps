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

GSM8K evaluation completed on all 1,319 prompts. Acceptance is the share
of proposals in the accepted prefix. The position rates are the share
of fully emitted rounds accepting at least that many proposals.

| Drafter | Tau | Acceptance | Decode tok/s | Speedup on AR-matched outputs |
| --- | ---: | ---: | ---: | ---: |
| Scratch, step 5,000 | 3.648 | 17.65% | 74.79 | 2.76x |
| Scratch, step 10,000 | 4.143 | 20.95% | 87.93 | 3.17x |
| Scratch, step 11,652 | 4.161 | 21.07% | 85.34 | 3.16x |
| Published b16, final checkpoint's job | 5.972 | 33.15% | 122.15 | 4.55x |

AR throughput ranged from 27.75 to 28.37 tok/s across the three jobs.
Published b16 ranged from 122.15 to 126.41 tok/s with identical acceptance
counts on every job. Decode throughput excludes prefill and the first token.

| Drafter | First | Second | Fourth | Eighth | Fifteenth |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scratch, step 5,000 | 80.31% | 59.64% | 29.89% | 5.13% | 0.01% |
| Scratch, step 10,000 | 82.94% | 64.06% | 36.04% | 9.31% | 0.18% |
| Scratch, step 11,652 | 83.00% | 64.30% | 36.24% | 9.48% | 0.19% |
| Published b16 | 86.97% | 72.49% | 50.15% | 25.45% | 7.33% |

The three trained checkpoints and published b16 produced identical sequences
on all prompts. Those sequences match sequential AR on the same 695 prompts
across all jobs. Speedup versus AR uses this common set. The raw reports
contain all 15 position rates and prompt-level measurements. Use
`python -m scripts.dflash_repro --phase eval --step <step> --report`.

- Step 5,000 — `lm-mpi-job-0eb55b38-de93-42d0-af7e-d516772c9ec3`.
- Step 10,000 — `lm-mpi-job-f1e7ad53-c0df-40b8-aa8d-1c9c7653cca3`.
- Step 11,652 — `lm-mpi-job-fe89f3cb-efbe-4efa-a3e5-9f3ea6f49a4d`.

## Conclusions

The full run took approximately as long as the probe projected.
Most of the measured gain in accepted length came between the first two
checkpoints. From step 10,000 to 11,652, tau rose by 0.019. At the final
checkpoint the first proposal was within four percentage points of the
published drafter, while the tail of the block was accepted much less often.
Throughput relative to the same-job published b16 was 69.6% at step 10,000
and 69.9% at step 11,652. Greedy agreement with sequential AR was partial.

## Changelog

- 2026-11-20 — Added the standalone reproduction and timed distributed probe.
- 2026-11-20 — Recorded probe timings and the full training estimate.
- 2026-11-21 — Verified completed training and all three checkpoints.
- 2026-11-21 — Recorded the paired GSM8K results for all checkpoints.
