## Hypothesis

A continued AR model can retain its native next-token readout while learning
future prediction slots. Compare this recipe with the completed SPS-MTP recipe
on next-token quality, instruction following and absolute decoding throughput.

## Setup

Training function for arm A is `scripts.real_readout_train.main`. Arm B reuses
the completed `scripts.vllm_sps_train.main` run.

## Results

Before MTP, held-out NLL is 0.3810 for the continued AR parent and
0.3964 for the SPS parent on the same 34,423 supervised tokens. Endpoint
comparison is pending. The probe passed the native-readout audit on the tested
prefix before and after frozen-stage updates. The BF16 argmax agreed at every
tested position, with maximum total variation 0.03153. The two stages took
5.14 and 7.49 seconds per measured worst-case step respectively.

## Conclusions

Measure both parents before MTP training and both final models with sequential
and block decoding. Report parent-to-final changes alongside paired endpoint
differences. The parents underwent different adaptation, and removing the first
slot also removes a state visible to later slots. The comparison therefore
estimates the effect of the complete recipes. A same-parent, identical-layout
readout control would isolate the choice of next-token readout.

## Changelog

- 2026-11-25: Planned staged training of the native-readout arm and comparison
  against the completed SPS-MTP arm.
- 2026-11-25: Submitted arm A probe `lm-mpi-job-4bb134b0-7807-4b40-abf0-dc0c64165658`.
- 2026-11-25: Probe completed; submitted arm A training as `lm-mpi-job-431afba4-ec39-4582-af0f-3b7751d85ad1`.
