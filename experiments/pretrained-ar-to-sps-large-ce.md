# Hypothesis

Larger pretrained SmolLM2 models can recover source-model quality under staged zero-window SPS adaptation using cross-entropy alone.

# Setup

`scripts.ar_sps_train.main`, selected through the larger CE registries in `scripts/ar_sps_config.py`. Each suite includes its own original-AR baseline and staged SPS arm. Preparation refuses to reduce the prescribed training exposure. Data preparation uses each pinned source tokenizer and isolated artifacts.

# Results

SmolLM2-360M evaluation completed. Final SPS versus original AR accuracy percentages:
ARC-Easy 67.09 vs 68.18; HellaSwag 56.38 vs 56.40; PIQA 71.60 vs 71.98;
SciQ 84.60 vs 85.90; LAMBADA 51.81 vs 53.02. Held-out NLL rises from
2.605900 to 2.617682. The observed-retention criterion fails. Final and
validation-selected best use the same endpoint. Per-task paired intervals exclude
zero degradation for ARC-Easy, SciQ and LAMBADA; these are not simultaneous
intervals and the experiment has one seed. Full results are in the shared artifact
root under `ar-sps-360m-ce/final-report.json` and `final-metrics.csv`.

SmolLM2-1.7B evaluation completed. Final SPS versus original AR accuracy percentages:
ARC-Easy 73.32 vs 73.15; HellaSwag 71.37 vs 71.48; PIQA 77.48 vs 77.58;
SciQ 89.90 vs 90.80; LAMBADA 66.02 vs 67.32. Held-out NLL rises from
2.305988 to 2.326675. Strict observed retention fails. Final and validation-selected
best use the same endpoint. Individual paired intervals exclude zero for SciQ and
LAMBADA degradation and the NLL increase; the other benchmark intervals include
zero. Full results are under `ar-sps-1.7b-ce/final-report.json` and
`final-metrics.csv` in the shared artifact root.

The embedding-only 1.7B endpoint exceeds source accuracy on ARC-Easy, SciQ and
LAMBADA, but has worse HellaSwag, PIQA and held-out NLL; it also fails retention.

# Conclusions

Neither larger-model run preserves original-model quality under the strict observed criterion. Increasing model size alone did not resolve retention in this pilot. These single-seed results do not establish that recovery is impossible with another training protocol. These runs do not test whether the embedding-only warmup outperforms immediate full training, or separate adaptation effects from ordinary continued training.

# Changelog

- 2026-11-01: Added isolated larger-model CE experiments using the existing preparation, training, evaluation and reporting pipeline.

- 2026-11-01: Recorded completed 360M evaluation; strict retention failed.

- 2026-11-01: Recorded completed 1.7B evaluation; both larger CE runs failed strict retention.
