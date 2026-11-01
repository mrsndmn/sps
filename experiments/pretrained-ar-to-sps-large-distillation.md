# Hypothesis

Frozen original-AR teacher distillation improves retention relative to staged CE for larger pretrained SmolLM2 models.

# Setup

`scripts.ar_sps_train.main` with the larger KD registries in `scripts/ar_sps_config.py`. Each fresh student uses its own frozen original teacher and the exact completed CE corpus, order and stage endpoints. Model identities, objectives, reference signatures, controls and budgets are defined in code. Artifacts are isolated from completed CE runs.

# Results

Pending preparation and execution.

# Conclusions

Compare each KD endpoint against its same-size original AR and staged CE control. No benchmark-based checkpoint selection; strict observed retention remains the success criterion.

# Changelog

- 2026-11-01: Added larger-model two-stage distillation experiments.
