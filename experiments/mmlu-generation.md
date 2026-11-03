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

# Conclusions

SmolLM2-1.7B meets the observed no-degradation criterion for full-MMLU overall
accuracy with both adaptations, by just two additional correct answers for CE
and six for CE+KD. This does not establish statistical superiority or retention
on every subject; the previously completed STEM subset declines for both arms.
The experiment uses one seed. Inspect invalid answers and truncation alongside
accuracy when interpreting reasoning quality.

# Changelog

- 2026-11-02: Added resumable reasoning-generation evaluation for source and adapted final checkpoints.

- 2026-11-02: Added full-MMLU continuation with validated reuse of full and STEM answers and an explicitly excluded slow arm in the code registry.

- 2026-11-03: Recorded completed SmolLM2-1.7B full-MMLU results and validated complete dataset coverage.
