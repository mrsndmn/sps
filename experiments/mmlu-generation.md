# Hypothesis

Zero-window SPS adaptation can preserve the source model's accuracy when MMLU answers are generated with reasoning. CE and CE+KD are compared with the same source model.

# Setup

`scripts.mmlu_run.evaluate`; executable protocol and checkpoint registry in `scripts/mmlu_config.py`, prompt, generation and answer extraction in `scripts/mmlu_generation.py`.

# Results

SmolLM2-1.7B completed the full test set. Source AR, final CE and final CE+KD
accuracy is 46.13%, 46.15% and 46.18% (6478, 6480 and 6484 correct answers).
Invalid-answer rates are 6.58%, 6.40% and 5.81%; generation-limit rates are
5.26%, 5.88% and 5.06%. No context overflows occurred. All runs cover the same
14042 unique document IDs; hashes, subjects and targets match the pinned dataset.
Summary artifacts: `mmlu-full-resume/reports/smol1.7.json` and `smol1.7.csv`.

This custom generative protocol measures both question answering and adherence
to the requested answer format. It is not the likelihood-based MMLU score.

SmolLM2-360M completed the full test set. Source AR, final CE and final CE+KD
accuracy is 22.96%, 23.34% and 22.70% (3224, 3278 and 3187 correct answers).
Invalid-answer rates are 8.04%, 6.27% and 8.13%; generation-limit rates are
5.70%, 5.93% and 6.45%. No context overflows occurred. All runs cover the same
14042 unique document IDs; hashes, subjects and targets match the pinned dataset.
CE gains 54 correct answers versus source (0.38 percentage points); KD loses
37 (0.26 points) and trails CE by 91 answers (0.65 points).
Summary artifacts: `mmlu-full-resume/reports/smol360.json` and `smol360.csv`.

Qwen3-0.6B source AR and final CE+KD completed the full test set. Accuracy is
46.09% and 45.06% (6472 and 6327 correct answers). Invalid-answer rates are
5.16% and 5.91%; generation-limit rates are 3.33% and 5.16%. No context overflows
occurred. Both runs cover the same 14042 unique document IDs; hashes, subjects
and targets match the pinned dataset. KD loses 145 correct answers versus
source (1.03 percentage points). Plain CE was explicitly excluded from this
full-MMLU continuation and has no complete score in this comparison.
Summary artifacts: `mmlu-full-resume/reports/qwen0.6.json` and `qwen0.6.csv`.

Qwen3-1.7B completed the full test set. Source AR, final CE and final CE+KD
accuracy is 56.73%, 52.63% and 54.98% (7966, 7390 and 7720 correct answers).
Invalid-answer rates are 3.43%, 11.49% and 6.87%; generation-limit rates are
2.76%, 9.66% and 6.52%. No context overflows occurred. All runs cover the same
14042 unique document IDs; hashes, subjects and targets match the pinned dataset.
CE loses 576 correct answers versus source (4.10 percentage points); KD loses
246 (1.75 points). KD recovers 330 answers over CE (2.35 points).
Summary artifacts: `mmlu-full-resume/reports/qwen1.7.json` and `qwen1.7.csv`.

# Conclusions

SmolLM2-1.7B meets the observed no-degradation criterion for full-MMLU overall
accuracy with both adaptations, by just two additional correct answers for CE
and six for CE+KD. This does not establish statistical superiority or retention
on every subject; the previously completed STEM subset declines for both arms.
SmolLM2-360M meets observed full-MMLU accuracy retention with CE but not CE+KD, although both had exceeded source accuracy on STEM alone. Qwen3-0.6B CE+KD fails observed full-MMLU retention, with a 1.03-point accuracy drop relative to source. Qwen3-1.7B KD recovers part of the CE degradation and reduces invalid answers, but both adaptations fail observed full-MMLU accuracy retention. Its near-restoration on STEM does not extend to the full benchmark. The experiment uses one seed. Inspect invalid answers and truncation alongside
accuracy when interpreting reasoning quality.

# Changelog

- 2026-11-02: Added resumable reasoning-generation evaluation for source and adapted final checkpoints.

- 2026-11-02: Added full-MMLU continuation with validated reuse of full and STEM answers and an explicitly excluded slow arm in the code registry.

- 2026-11-03: Recorded completed SmolLM2-1.7B full-MMLU results and validated complete dataset coverage.

- 2026-11-03: Recorded completed SmolLM2-360M full-MMLU results and validated complete dataset coverage.

- 2026-11-03: Recorded completed Qwen3-0.6B base/KD full-MMLU results and validated complete dataset coverage.

- 2026-11-03: Recorded completed Qwen3-1.7B full-MMLU results and validated complete dataset coverage.
