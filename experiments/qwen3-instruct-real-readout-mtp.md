## Hypothesis

A continued AR model can retain its native next-token readout while learning
future prediction slots. Compare this recipe with the completed SPS-MTP recipe
on next-token quality, instruction following and absolute decoding throughput.

## Setup

Training function for arm A is `scripts.real_readout_train.main`. Arm B reuses
the completed `scripts.vllm_sps_train.main` run.

## Results

Before MTP, held-out NLL is 0.3810 for the continued AR parent and
0.3964 for the SPS parent on the same 34,423 supervised tokens. At the final MTP checkpoint the corresponding values are 0.4366 and
0.4510, increases of 0.0555 and 0.0546. Packed held-out CE for horizons two
through four is 1.2454/1.9471/2.5542 for A and 1.2444/1.9529/2.5562 for B.
Instruction following and timing are pending. The probe passed the native-readout audit on the tested
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
- 2026-11-25: Training reached its fixed final endpoint. Aggregate validation CE across horizons was 1.5965 for A and 1.6030 for B.

- 2026-11-25: Submitted one-GPU IFEval for A non-thinking (`lm-mpi-job-81d30d24-1da3-46f1-ae4a-b1c9e0c6224f`) and thinking (`lm-mpi-job-c5b3180f-f6d5-43a4-86ef-4ee9fefdea6f`), and B non-thinking (`lm-mpi-job-1ef8fc08-14f8-4e45-ab7c-6a3c57b96e80`) and thinking (`lm-mpi-job-c2a37cc9-3ef3-43f3-9caf-bb748f741f55`). Checked that parent and endpoint held-out token counts and example IDs match.
- 2026-11-25: Submitted one-GPU matched serving audits and timings for A (`lm-mpi-job-befe903a-8394-4c95-9046-2a11690b3435`) and B (`lm-mpi-job-3dcb62fd-27f7-4d57-be9a-89ce0ddd89d1`).
- 2026-11-25: Submitted matched zero-shot likelihood evaluations against both parents' saved task results, A `lm-mpi-job-3643aa2b-d6ed-4dd0-abcb-17f14b5d7cd9` and B `lm-mpi-job-f43fd8ba-67f1-433d-bd14-42a0936d78d3`.
- 2026-11-25: Stopped both likelihood benchmark jobs at the user's request. Added paired generative GSM8K evaluation.
- 2026-11-25: Submitted GSM8K chat generation in thinking/non-thinking modes for A (`lm-mpi-job-ebeaed92-0da3-4271-8d32-a1cad465997e`, `lm-mpi-job-4a244d43-e261-45e9-8654-da815429a230`) and B (`lm-mpi-job-e3641317-5874-45cf-bc89-d8a06b4697d1`, `lm-mpi-job-2625b364-a4dd-400c-b289-f0066074440b`).
- 2026-11-25: Submitted full generative MMLU in thinking and non-thinking modes for A (`lm-mpi-job-8f7d1a1b-5ba5-443f-848b-f12e40b5fd59`, `lm-mpi-job-7696b8ca-a387-4892-9bac-02cd79858b75`) and B (`lm-mpi-job-3a6d6a5d-1fc5-411d-97eb-903d2f781ac8`, `lm-mpi-job-580bd14a-9a2f-4266-8e92-d5ced0c44992`).
