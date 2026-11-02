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

Qwen3-0.6B KD evaluation completed. Final accuracy percentages: ARC-Easy 62.88,
HellaSwag 52.99, PIQA 70.08, SciQ 90.40, LAMBADA 51.99; held-out NLL 2.761109.
Against matched final CE, KD improves HellaSwag by 1.14 percentage points, SciQ
by 1.20, LAMBADA by 1.73 and NLL by 0.020507; their individual paired 95%
intervals exclude zero. ARC-Easy falls by 0.88 points, with an interval including
zero; PIQA's interval also includes zero. Original AR retention still fails at
all evaluated KD checkpoints. Best equals final. Embedding-only KD reaches
LAMBADA 54.61%, declining to 51.99% after full unfreezing.
Complete size-specific results: `ar-sps-qwen3-0.6b/final-report.json` and
`final-metrics.csv` under the shared artifact root.

Both sizes have completed CE/KD training. Qwen3-1.7B evaluations remain pending.

# Conclusions

The completed 0.6B CE run shows mixed benchmark changes and fails retention, despite a substantial ARC-Easy gain. At matched final exposure, KD recovers part of the CE degradation in several tasks and held-out NLL, but still does not preserve original-model quality. Results use one seed and per-task intervals without simultaneous correction. This matrix does not separate continued-training effects from SPS adaptation; no AR-continuation arm is included in the requested scope. Cross-family per-token losses are not directly comparable across tokenizers.

# Changelog

- 2026-11-01: Added paired Qwen3 CE/KD suites and native-backbone correctness gates.

- 2026-11-02: Recorded Qwen3-0.6B CE results; strict retention failed.

- 2026-11-02: Recorded Qwen3-0.6B KD and paired CE comparisons; strict retention failed.
