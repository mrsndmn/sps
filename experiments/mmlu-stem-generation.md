# Hypothesis

STEM-specific accuracy can reveal differences between the source model and CE/KD SPS adaptations that are hidden by the overall MMLU score.

# Setup

`scripts.mmlu_run.evaluate` runs the STEM-only suite with validated reuse of previous answers. `scripts.report_mmlu_stem.main` aggregates the existing generative evaluation using the official MMLU category mapping. Its source protocol is defined in `scripts/mmlu_config.py` and `scripts/mmlu_generation.py`.

# Results

Pending. The report exposes coverage while evaluation is running and publishes accuracy only when every STEM question has an answer record.

# Conclusions

Compare source, CE and CE+KD within each model. Answer-format failures remain errors under the original generative protocol.

# Changelog

- 2026-11-02: Added a separate STEM report reusing existing generated answers.

- 2026-11-02: Added STEM-only generation to replace full-MMLU jobs while retaining existing STEM answers.
