## Hypothesis

A continued AR model can retain its native next-token readout while learning
future prediction slots. Compare this recipe with the completed SPS-MTP recipe
on next-token quality, instruction following and absolute decoding throughput.

## Setup

Training function for arm A is `scripts.real_readout_train.main`. Arm B reuses
the completed `scripts.serving.vllm_sps_train.main` run.

## Results

Before MTP, held-out NLL is 0.3810 for the continued AR parent and
0.3964 for the SPS parent on the same 34,423 supervised tokens. At the final MTP checkpoint the corresponding values are 0.4366 and
0.4510, increases of 0.0555 and 0.0546. Packed held-out CE for horizons two
through four is 1.2454/1.9471/2.5542 for A and 1.2444/1.9529/2.5562 for B.
The probe passed the native-readout audit on the tested
prefix before and after frozen-stage updates. The BF16 argmax agreed at every
tested position, with maximum total variation 0.03153. The two stages took
5.14 and 7.49 seconds per measured worst-case step respectively.

On the 100-question MMLU STEM pilot, median single-request decode throughput
in tokens/s and correct answers per 100 were as follows. Each pair gives
sequential then block decoding on the same final checkpoint.

| Readout | Thinking tok/s | Non-thinking tok/s | Thinking correct | Non-thinking correct |
|---|---:|---:|---:|---:|
| A, native | 193.4 → 262.3 | 198.3 → 336.2 | 60 → 59 | 57 → 58 |
| B, SPS first slot | 161.4 → 255.0 | 185.3 → 335.8 | 70 → 63 | 55 → 59 |

The two decoders agreed on all 16 greedy tokens for 8/8 prefixes with A and
7/8 with B. B first diverged on the fifth generated token of one prefix.
CUDA graph replay was verified during warmup. For thinking and non-thinking
respectively, A averaged 2.19 and 2.55 tokens per verification round, with
median decode time per round of 8.33 and 7.38 ms. B averaged 2.17 and 2.58
tokens per round, taking 8.48 and 7.42 ms.

HumanEval+ generation produced one record per task in all four runs. In thinking mode, A and B reached the length cap on 45 and 49 tasks, with 37 and 42 empty extracted solutions respectively. In non-thinking mode, A reached the cap once and B never reached it. Both exported 164 nonempty solutions. External EvalPlus scoring covered all 164 tasks in each run. In thinking mode, A and B passed 66 and 66 base tasks and 59 and 63 tasks with both base and extra tests. In non-thinking mode, the corresponding base counts were 75 and 69, with 67 and 65 passing both suites.

All four sequential-generation IFEval runs covered the same 541 prompts. In
thinking mode, strict prompt passes changed from 360 to 309 for A and 349 to
293 for B after MTP. In non-thinking mode, they changed from 285 to 272 for A
and 265 to 271 for B. The thinking-mode rates of malformed reasoning boundaries
rose from 4.25% to 12.57% for A and from 6.84% to 12.01% for B.

Full zero-shot generative MMLU covered all 14,042 questions and 57 subjects.
In thinking mode, A and B scored 59.42% and 60.38% (subject macro 61.44%
and 62.63%). In non-thinking mode they scored 53.62% and 53.28% (macro
55.42% and 55.04%). On the 3,018 STEM questions, thinking accuracy was
64.81% and 66.34%, and both scored 54.37% in non-thinking mode.
Unparseable answers counted as incorrect. Their thinking rates were
13.83% and 12.36%, and non-thinking rates were 1.90% and 3.82%.

On all 1,319 GSM8K questions, strict numeric accuracy for A/B was 36.09%/32.90% in thinking and 42.76%/2.35% in non-thinking. The parser required a final line containing the requested hash delimiter and number. In B non-thinking, 1,262 outputs were unparseable and seven reached the generation cap. An exploratory fallback accepting the last simple boxed number in otherwise unparseable final text changed thinking scores to 71.11%/73.24% and non-thinking scores to 57.54%/14.86%. These additional scores depend on the extraction rule.

## Conclusions

Measure both parents before MTP training and both final models with sequential
and block decoding. Report parent-to-final changes alongside paired endpoint
differences. The parents underwent different adaptation, and removing the first
slot also removes a state visible to later slots. The comparison therefore
estimates the effect of the complete recipes. A same-parent, identical-layout
readout control would isolate the choice of next-token readout.

On this pilot, A's native readout supports block decoding at 1.36× its
sequential throughput in thinking mode and 1.70× in non-thinking mode.
B's larger relative gains coincide with a slower sequential baseline and
its timings are diagnostic given the greedy divergence. Absolute block
throughput was close between recipes, especially in non-thinking mode.

Thinking-mode strict accuracy declined for both recipes. The difference in
their parent-to-final changes (A minus B) was +0.92 percentage points in
thinking mode and −3.51 points in non-thinking mode. Paired 95% percentile
bootstrap intervals over prompts were [−4.07, +5.91] and [−8.69, +1.48]
respectively. This comparison does not resolve a readout advantage.

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
- 2026-11-25: Added HumanEval+ generation with separate raw responses and portable EvalPlus samples for external correctness scoring.
- 2026-11-25: Submitted HumanEval+ generation in thinking/non-thinking modes for A (`lm-mpi-job-c8988d36-8e81-4f4e-97c8-84ee2ae945f1`, `lm-mpi-job-ec501146-edb6-4d7a-88bc-cf41da9c175a`) and B (`lm-mpi-job-6dc8e887-37d6-4e51-bd7c-04fea0dd54f6`, `lm-mpi-job-3f623dd1-365c-4f93-bcfa-a2447273ca71`).
- 2026-11-25: Verified both serving runs against their submitted code and recorded paired throughput, STEM accuracy and greedy agreement.
- 2026-11-26: Verified task coverage, sample extraction and raw/export hashes for all four HumanEval+ runs and packaged the samples for external scoring.
- 2026-11-26: Verified all eight parent and final IFEval answer sets against the pinned prompts, reproduced per-answer scores, and compared paired changes.
- 2026-11-26: Scored all four HumanEval files on the external VM and verified returned solutions and task IDs against the submitted files.
- 2026-11-26: Verified four full-MMLU checkpoint manifests, answer hashes and per-question scores against the pinned test split.
- 2026-11-26: Verified all four GSM8K answer files, seeds and strict scores, and measured sensitivity to boxed-answer formatting.
