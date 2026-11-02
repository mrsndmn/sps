# Hypothesis

Frozen original-AR teacher distillation improves retention relative to staged CE for larger pretrained SmolLM2 models.

# Setup

`scripts.ar_sps_train.main` with the larger KD registries in `scripts/ar_sps_config.py`. Each fresh student uses its own frozen original teacher and the exact completed CE corpus, order and stage endpoints. Model identities, objectives, reference signatures, controls and budgets are defined in code. Artifacts are isolated from completed CE runs.

# Results

SmolLM2-360M completed. Final KD accuracy percentages are ARC-Easy 67.51,
HellaSwag 56.12, PIQA 71.93, SciQ 85.90 and LAMBADA 52.30. Held-out NLL is
2.614695 versus staged CE 2.617682 and original AR 2.605900. At matched final
exposure, KD improves SciQ over CE by 1.30 percentage points (individual paired
95% interval [0.50, 2.10]) and NLL by 0.002987 (KD minus CE interval
[-0.004046, -0.001955]). HellaSwag decreases; other benchmark intervals include
or touch zero. Final and validation-selected best coincide. All evaluated KD
checkpoints fail strict observed retention relative to original AR.

Artifacts: `ar-sps-360m-kd/final-report.json` and `final-metrics.csv` under the
shared artifact root.

SmolLM2-1.7B completed. Final KD accuracy percentages are ARC-Easy 73.48,
HellaSwag 71.40, PIQA 77.37, SciQ 90.00 and LAMBADA 66.49. Held-out NLL is
2.321363 versus staged CE 2.326675 and original AR 2.305988. KD minus CE NLL
has paired 95% interval [-0.006324, -0.004342]; every benchmark difference interval
includes or touches zero. Relative to original AR, SciQ and LAMBADA remain lower
with individual paired intervals excluding zero. All evaluated KD checkpoints fail
strict retention. Final and validation-selected best coincide. The embedding-only
endpoint reaches LAMBADA 68.33%, declining to 66.49% after full unfreezing.
Artifacts are under `ar-sps-1.7b-kd/final-report.json` and `final-metrics.csv`.

# Conclusions

On 360M, distillation partially recovers CE losses, including matching original SciQ, but does not preserve all original-model metrics. Results are from one seed and intervals are per-task, without simultaneous correction. On 1.7B, KD also improves held-out loss over CE but fails original-model retention. Across all tested SmolLM2 sizes, this distillation recipe reduces final NLL relative to staged CE without recovering every source metric. No new experiment is launched from this outcome.

# Changelog

- 2026-11-01: Added larger-model two-stage distillation experiments.

- 2026-11-02: Recorded 360M KD evaluation and matched CE comparison; retention failed.

- 2026-11-02: Recorded 1.7B KD evaluation; both larger KD runs failed strict retention.
