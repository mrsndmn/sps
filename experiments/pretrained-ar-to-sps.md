# Hypothesis
A pretrained autoregressive model can recover its measured benchmark quality after conversion to zero-window SPS. Learning the prediction embedding before unfreezing the backbone may improve recovery relative to immediate full adaptation.

# Setup
Training function: `scripts.ar_sps_train.main`. The `ar-sps` suite selects the executable experiment registry. Source identity, training arguments, job resources, budget and scoring rules are defined in code.

# Results
All training and evaluation jobs completed. The two trained SPS variants and
AR continuation were compared at matched final exposure. Embedding-only is an
intermediate checkpoint, not an equal-budget final treatment. Values are the
configured primary benchmark percentages and held-out NLL in nat/token.

| Variant | ARC-Easy | HellaSwag | PIQA | SciQ | LAMBADA | NLL |
|---|---:|---:|---:|---:|---:|---:|
| Original AR | 58.63 | 43.07 | 68.39 | 78.40 | 42.71 | 2.828 |
| Untrained SPS | 46.89 | 34.18 | 60.61 | 62.30 | 0.76 | 6.165 |
| Embedding-only SPS | 56.52 | 41.65 | 65.23 | 76.50 | 42.81 | 3.049 |
| Two-stage SPS | 57.66 | 43.27 | 68.28 | 77.30 | 41.90 | 2.857 |
| Direct full SPS | 57.45 | 43.22 | 67.90 | 76.50 | 41.10 | 2.863 |
| Continued AR | 57.37 | 43.59 | 68.50 | 77.10 | 43.12 | 2.818 |

Validation-selected best checkpoints coincide with the final checkpoint for both
SPS treatments and with the intermediate checkpoint for AR continuation. The
latter scores 58.54 / 43.54 / 68.12 / 77.20 / 43.02 with NLL 2.817; its lower
exposure must not be used for matched-final causal claims.

Two-stage SPS versus original AR changes the benchmark scores by -0.97, +0.20,
-0.11, -1.10 and -0.82 percentage points, respectively, and increases NLL by
0.029. Its LAMBADA paired 95% delta interval is [-1.59, -0.02] percentage points.
Two-stage SPS versus direct full SPS improves all five point estimates; only
the LAMBADA benchmark interval excludes zero: +0.80 points [0.29, 1.30]. NLL
improves by 0.006 [0.00495, 0.00710]. These are per-metric intervals, not
simultaneous guarantees or training-seed uncertainty.

Full paired outputs: `sps-artifacts/ar-sps/final-report.json`; summary:
`sps-artifacts/ar-sps/final-metrics.csv` under the project artifact parent.

# Conclusions
The strict observed-quality-retention criterion is not met by either SPS
adaptation. Embedding-only training repairs most of the initial collapse but
leaves broad deficits. Full adaptation gets close to the original AR model;
embedding warmup modestly improves the final result over immediate full tuning.
Ordinary AR continuation also loses some benchmark quality, so the experiment
does not attribute every benchmark decline solely to SPS conversion. The
remaining LAMBADA/NLL gap versus matched AR continuation warrants further work.
A proposed next experiment is source-AR distillation during adaptation, with
validation-only selection and additional seeds to check robustness. No further
runs are launched by this result update. This single-model pilot does not prove
conversion without quality loss impossible.

# Changelog
- 2026-10-28: Added the interviewed adaptation design and bounded launch/evaluation paths.

- 2026-10-29: Separated production-dtype same-shape causality checks from the FP64 cross-length reference check; original tolerances retained.

- 2026-10-30: Recorded completed baseline evaluations; untrained conversion loses quality on every selected benchmark.

- 2026-10-30: Recorded all final and intermediate evaluations, matched-exposure controls and the failed strict retention criterion.
