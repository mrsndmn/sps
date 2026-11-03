# Hypothesis

STEM-specific accuracy can reveal differences between the source model and CE/KD SPS adaptations that are hidden by the overall MMLU score.

# Setup

`scripts.mmlu_run.evaluate` runs the STEM-only suite with validated reuse of previous answers. `scripts.report_mmlu_stem.main` aggregates the existing generative evaluation using the official MMLU category mapping. Its source protocol is defined in `scripts/mmlu_config.py` and `scripts/mmlu_generation.py`.

# Results

SmolLM2-1.7B completed all STEM questions for source AR, final CE and final CE+KD.
Accuracy is 44.37%, 43.24% and 43.34%, respectively. Invalid-answer rates are
6.79%, 8.32% and 7.09%; generation-limit rates are 5.80%, 7.85% and 6.39%.
No context overflows occurred. All three runs cover identical document IDs and
hashes. CE loses 1.13 percentage points versus source, and CE+KD loses 1.03.
CE+KD exceeds CE by only three correct answers (0.10 points).
The shared artifact report is `mmlu-stem-generation/reports/mmlu_stem.json`.
All model groups have now completed; the consolidated report contains all final scores.

SmolLM2-360M completed all STEM questions with paired document IDs and hashes.
Source AR, final CE and final CE+KD accuracy is 22.03%, 22.86% and 22.53%
(665, 690 and 680 correct answers). Invalid-answer rates are 8.75%, 6.79% and
8.25%; generation-limit rates are 6.53%, 6.43% and 6.73%. No context overflows
occurred. CE gains 0.83 percentage points versus source and CE+KD gains 0.50;
KD trails CE by ten correct answers (0.33 points).

SmolLM2-135M completed all STEM questions with paired document IDs and hashes.
Source AR, final CE and final CE+KD accuracy is 22.70%, 19.65% and 20.58%
(685, 593 and 621 correct answers). Invalid-answer rates are 17.13%, 23.89%
and 25.15%; generation-limit rates are 6.53%, 17.66% and 13.19%. No context
overflows occurred. CE loses 3.05 percentage points versus source and CE+KD
loses 2.12. KD exceeds CE by 28 correct answers (0.93 points), despite a higher
invalid-answer rate.

Qwen3-1.7B completed all STEM questions with paired document IDs and hashes.
Source AR, final CE and final CE+KD accuracy is 61.23%, 57.46% and 61.07%
(1848, 1734 and 1843 correct answers). Invalid-answer rates are 3.11%, 9.77%
and 5.43%; generation-limit rates are 2.45%, 8.35% and 5.07%. No context
overflows occurred. CE loses 3.78 percentage points versus source; KD loses
0.17 points, or five correct answers. KD exceeds CE by 109 correct answers
(3.61 points).

Qwen3-0.6B completed all STEM questions with paired document IDs and hashes.
Source AR, final CE and final CE+KD accuracy is 50.43%, 5.47% and 49.04%
(1522, 165 and 1480 correct answers). Invalid-answer rates are 4.84%, 88.24%
and 4.94%; generation-limit rates are 3.02%, 89.40% and 3.88%. No context
overflows occurred. CE loses 44.96 percentage points versus source, and KD
loses 1.39 points (42 answers). The very low CE score coincides with pervasive
format failures and generation-limit stops; it cannot be interpreted as a
measurement of knowledge alone. The pinned-dataset report validated all final
STEM runs; consolidated CSV: `mmlu-stem-generation/reports/mmlu_stem.csv`.

# Conclusions

SmolLM2-1.7B fails the observed no-degradation criterion for both adapted final checkpoints. KD reduces format failures relative to CE but has nearly identical accuracy. SmolLM2-360M meets the no-degradation criterion for this benchmark’s measured overall accuracy with both adaptations; this does not imply retention across other benchmarks. SmolLM2-135M fails observed retention for both adaptations; KD recovers some accuracy relative to CE but increases invalid answers. Qwen3-1.7B KD nearly restores source accuracy and substantially exceeds CE in measured accuracy, but still fails the strict observed no-degradation criterion by five answers. Qwen3-0.6B CE has severe answer-format failures under this protocol; KD largely avoids these failures but still falls below source accuracy. All requested STEM evaluations are complete. This is one seed and no statistical superiority claim is made. Answer-format failures remain errors under the original generative protocol.

# Changelog

- 2026-11-02: Added a separate STEM report reusing existing generated answers.

- 2026-11-02: Added STEM-only generation to replace full-MMLU jobs while retaining existing STEM answers.

- 2026-11-02: Recorded completed SmolLM2-1.7B STEM results and validated paired document coverage.

- 2026-11-02: Recorded completed SmolLM2-360M STEM results and paired coverage checks.

- 2026-11-02: Recorded completed SmolLM2-135M STEM results and paired coverage checks.

- 2026-11-02: Recorded completed Qwen3-1.7B STEM results and paired coverage checks.

- 2026-11-03: Recorded completed Qwen3-0.6B STEM results and finalized the consolidated report for every model.
