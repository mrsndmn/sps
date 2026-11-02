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
Other model groups are tracked independently; partial groups have no final score.

SmolLM2-360M completed all STEM questions with paired document IDs and hashes.
Source AR, final CE and final CE+KD accuracy is 22.03%, 22.86% and 22.53%
(665, 690 and 680 correct answers). Invalid-answer rates are 8.75%, 6.79% and
8.25%; generation-limit rates are 6.53%, 6.43% and 6.73%. No context overflows
occurred. CE gains 0.83 percentage points versus source and CE+KD gains 0.50;
KD trails CE by ten correct answers (0.33 points).

# Conclusions

SmolLM2-1.7B fails the observed no-degradation criterion for both adapted final checkpoints. KD reduces format failures relative to CE but has nearly identical accuracy. SmolLM2-360M meets the no-degradation criterion for this benchmark’s measured overall accuracy with both adaptations; this does not imply retention across other benchmarks. This is one seed and no statistical superiority claim is made. Answer-format failures remain errors under the original generative protocol.

# Changelog

- 2026-11-02: Added a separate STEM report reusing existing generated answers.

- 2026-11-02: Added STEM-only generation to replace full-MMLU jobs while retaining existing STEM answers.

- 2026-11-02: Recorded completed SmolLM2-1.7B STEM results and validated paired document coverage.

- 2026-11-02: Recorded completed SmolLM2-360M STEM results and paired coverage checks.
