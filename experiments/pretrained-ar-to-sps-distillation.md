# Hypothesis
Matching a frozen original AR teacher during both stages of SPS adaptation will
reduce the remaining quality loss compared with the completed plain-CE treatment.

# Setup
Training function: `scripts.ar_sps_train.main`. The distillation suite selects its
registry and objective in executable code. The student starts from the pretrained
source; the completed CE treatment supplies immutable data and evaluation controls.

# Results
Completed. At matched final exposure, held-out NLL improves from staged CE's
2.856920 to 2.851728, but remains above original AR's 2.828018. The paired interval
for KD minus CE NLL is [-0.006165, -0.004199]. Final KD accuracy percentages are
ARC-Easy 57.91, HellaSwag 43.24, PIQA 68.23, SciQ 78.00 and LAMBADA 42.19.
Only HellaSwag exceeds original AR; all benchmark intervals for KD minus staged
CE include zero. Validation-selected best and final both select the last step.
Full paired results: `sps-artifacts/ar-sps-kd/final-report.json`; compact metrics:
`sps-artifacts/ar-sps-kd/final-metrics.csv` (artifact paths relative to the shared
project parent).

# Conclusions
The strict observed no-degradation criterion failed. Distillation modestly helps
held-out loss, but this single-seed pilot does not establish a benchmark advantage
over staged CE or recovery of original-model quality. No further distillation
sweep is launched; larger-model CE comparisons remain independent.

# Changelog
- 2026-11-01: Added the frozen-source-teacher distillation follow-up.

- 2026-11-01: Recorded completed distillation evaluation and paired comparisons.
