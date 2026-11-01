# Hypothesis

Two-stage zero-window SPS adaptation transfers to pretrained Qwen3, and frozen-source distillation improves retention relative to plain CE.

# Setup

`scripts.ar_sps_train.main` with the Qwen registries in `scripts/ar_sps_config.py`. Each suite prepares one native-tokenizer corpus shared by fresh staged CE and KD students, and includes its original AR baseline. Native Qwen3 layers retain Q/K normalization and configured attention dimensions. Only full-attention Qwen3 configurations are supported. Numeric design, pinned model revisions and budgets live in code.

# Results

Pending GPU preparation, including native parity, causality and both-arm throughput probes.

# Conclusions

Compare each adapted model against its own source and compare KD against CE at matched stage endpoints. This matrix does not separate continued-training effects from SPS adaptation; no AR-continuation arm is included in the requested scope. Cross-family per-token losses are not directly comparable across tokenizers.

# Changelog

- 2026-11-01: Added paired Qwen3 CE/KD suites and native-backbone correctness gates.
