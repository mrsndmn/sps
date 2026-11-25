## Hypothesis

A continued AR model can retain its native next-token readout while learning
future prediction slots. Compare this recipe with the completed SPS-MTP recipe
on next-token quality, instruction following and absolute decoding throughput.

## Setup

Training function for arm A is `scripts.real_readout_train.main`. Arm B reuses
the completed `scripts.vllm_sps_train.main` run.

## Results

Pending.

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
