# Hypothesis

Zero-window SPS adaptation can preserve the source model's accuracy when MMLU answers are generated with reasoning. CE and CE+KD are compared with the same source model.

# Setup

`scripts.mmlu_run.evaluate`; executable protocol and checkpoint registry in `scripts/mmlu_config.py`, prompt, generation and answer extraction in `scripts/mmlu_generation.py`.

# Results

Pending. This custom generative protocol measures both question answering and adherence to the requested answer format. It is not the likelihood-based MMLU score.

# Conclusions

Compare paired samples within each source model; inspect invalid answers and output truncation before attributing differences to reasoning quality.

# Changelog

- 2026-11-02: Added resumable reasoning-generation evaluation for source and adapted final checkpoints.
