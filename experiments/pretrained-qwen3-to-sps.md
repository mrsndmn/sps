# Hypothesis

Two-stage zero-window SPS adaptation transfers to pretrained Qwen3, and frozen-source distillation improves retention relative to plain CE.

# Setup

`scripts.ar_sps_train.main` with the Qwen registries in `scripts/ar_sps_config.py`. Each suite prepares one native-tokenizer corpus shared by fresh staged CE and KD students, and includes its original AR baseline. Native Qwen3 layers retain Q/K normalization and configured attention dimensions. Only full-attention Qwen3 configurations are supported. Numeric design, pinned model revisions and budgets live in code.

# Results

Qwen3-0.6B CE evaluation completed. Final SPS versus original AR accuracy percentages:
ARC-Easy 63.76 vs 58.08; HellaSwag 51.85 vs 53.74; PIQA 69.97 vs 69.75;
SciQ 89.20 vs 91.70; LAMBADA 50.26 vs 53.74. Held-out NLL rises from
2.754853 to 2.781615. Strict observed retention fails for every evaluated CE
checkpoint; validation-selected best equals final. Individual paired 95% intervals
exclude zero for the ARC improvement and HellaSwag, SciQ and LAMBADA losses,
but include zero for PIQA. Artifacts: `ar-sps-qwen3-0.6b/ce-report.json` and
`ce-metrics.csv` under the shared artifact root.

Both sizes have completed CE/KD training. Remaining evaluations are pending.

# Conclusions

The completed 0.6B CE run shows mixed benchmark changes and fails retention, despite a substantial ARC-Easy gain. Compare KD against CE at matched stage endpoints once its evaluation finishes. Results use one seed and per-task intervals without simultaneous correction. This matrix does not separate continued-training effects from SPS adaptation; no AR-continuation arm is included in the requested scope. Cross-family per-token losses are not directly comparable across tokenizers.

# Changelog

- 2026-11-01: Added paired Qwen3 CE/KD suites and native-backbone correctness gates.

- 2026-11-02: Recorded Qwen3-0.6B CE results; strict retention failed.
