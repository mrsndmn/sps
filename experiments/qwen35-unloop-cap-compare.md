## Hypothesis

The Unloop checkpoint can complete more thinking answers within the same generation budget and reduce repetitive tails compared with its original Qwen counterpart.

## Setup

Evaluation functions are `workloads.evaluation.qwen35_cap_compare.generate` and `workloads.evaluation.qwen35_cap_compare.summarize`.

## Results

Both runs completed on all 109 prompts. At a 2,048-token response limit, weighted completion was 29.7% for original and 40.0% for Unloop. Student-context fit was 25.8% and 37.6%, a paired difference of +11.8 percentage points (95% within-source paired bootstrap interval +4.5 to +19.7). At 8,192, completion reached 44.7% and 71.0%, and mean generated tokens were 5,499 and 4,013. The terminal repetition detector fired on 18 original traces and zero Unloop traces at this cap (weighted 6.4% and 0%).

## Conclusions

For either model, 2,048 is the smallest tested response budget recovering at least 95% of the student-encodable yield at 8,192. Unloop improved completion while using fewer tokens on this cohort of prompts previously capped by Qwen3-1.7B. Fit refers to the existing student's training encoding. Completion and tail repetition are structural measurements.

## Changelog

- 2026-11-26: Added matched fresh generations on the previous cap pilot prompts.
- 2026-11-26: Verified both pinned weight shards and submitted original `lm-mpi-job-a28ae99e-6376-42a2-8998-ed1e64869516` and Unloop `lm-mpi-job-49bf1921-f526-4260-a330-92c70428e239` to MLSpace.
- 2026-11-26: Verified 109 matched prompt hashes and seeds, raw outputs, encodings, and the weighted cap curves for both completed jobs.
