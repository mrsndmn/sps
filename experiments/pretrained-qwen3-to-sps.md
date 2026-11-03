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

Qwen3-1.7B CE evaluation completed. Final SPS versus original AR accuracy percentages:
ARC-Easy 73.57 vs 68.52; HellaSwag 64.67 vs 66.37; PIQA 74.81 vs 75.84;
SciQ 93.60 vs 94.90; LAMBADA 61.69 vs 63.09. Held-out NLL rises from
2.513415 to 2.531043. All evaluated CE checkpoints fail strict observed retention;
best equals final. Individual paired 95% intervals exclude zero for the ARC gain,
HellaSwag/SciQ/LAMBADA losses and NLL increase; PIQA includes zero. Artifacts:
`ar-sps-qwen3-1.7b/ce-report.json` and `ce-metrics.csv` in the shared artifact root.

Qwen3-1.7B KD evaluation completed. Final accuracy percentages: ARC-Easy 72.98,
HellaSwag 65.59, PIQA 75.30, SciQ 94.50, LAMBADA 62.25; held-out NLL 2.519932.
Compared with final CE, KD improves HellaSwag by 0.93 percentage points and SciQ
by 0.90; their individual paired 95% intervals exclude zero. NLL improves by
0.011112, also with an interval excluding zero. PIQA improves by 0.49 points,
LAMBADA by 0.56, and ARC falls by 0.59; those intervals include zero.
Strict observed retention versus original AR still fails for the final KD model.
Complete size-specific artifacts: `ar-sps-qwen3-1.7b/final-report.json` and
`final-metrics.csv`. Both sizes have completed CE/KD training and evaluation.

# Conclusions

Both sizes show mixed benchmark changes and fail final-checkpoint retention, despite substantial ARC-Easy gains. At matched final exposure, KD recovers part of the CE degradation in several tasks and held-out NLL, but still does not preserve original-model quality. Results use one seed and per-task intervals without simultaneous correction. This matrix does not separate continued-training effects from SPS adaptation; no AR-continuation arm is included in the requested scope. Cross-family per-token losses are not directly comparable across tokenizers.

# Changelog

- 2026-11-01: Added paired Qwen3 CE/KD suites and native-backbone correctness gates.

- 2026-11-02: Recorded Qwen3-0.6B CE results; strict retention failed.

- 2026-11-02: Recorded Qwen3-0.6B KD and paired CE comparisons; strict retention failed.

- 2026-11-02: Recorded Qwen3-1.7B CE evaluation; strict retention failed.

- 2026-11-02: Recorded Qwen3-1.7B KD results and paired CE comparisons; final strict retention failed.
