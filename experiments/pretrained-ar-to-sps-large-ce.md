# Hypothesis

Larger pretrained SmolLM2 models can recover source-model quality under staged zero-window SPS adaptation using cross-entropy alone.

# Setup

`scripts.ar_sps_train.main`, selected through the larger CE registries in `scripts/ar_sps_config.py`. Each suite includes its own original-AR baseline and staged SPS arm. Preparation refuses to reduce the prescribed training exposure. Data preparation uses each pinned source tokenizer and isolated artifacts.

# Results

Pending preparation and execution.

# Conclusions

Compare each adapted model against its own original baseline using the observed no-degradation criterion. These runs do not test whether the embedding-only warmup outperforms immediate full training.

# Changelog

- 2026-11-01: Added isolated larger-model CE experiments using the existing preparation, training, evaluation and reporting pipeline.
