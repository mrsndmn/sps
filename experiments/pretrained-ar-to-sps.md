# Hypothesis
A pretrained autoregressive model can recover its measured benchmark quality after conversion to zero-window SPS. Learning the prediction embedding before unfreezing the backbone may improve recovery relative to immediate full adaptation.

# Setup
Training function: `scripts.ar_sps_train.main`. The `ar-sps` suite selects the executable experiment registry. Source identity, training arguments, job resources, budget and scoring rules are defined in code.

# Results
Preparation and the revised numerical audit passed. Baseline evaluation of the
original AR versus the untrained SPS conversion gives ARC-Easy 58.63% vs 46.89%,
HellaSwag 43.07% vs 34.18%, PIQA 68.39% vs 60.61%, SciQ 78.40% vs 62.30%,
and LAMBADA 42.71% vs 0.76%. Held-out NLL increases from 2.828 to 6.165
nat/token. All paired benchmark delta intervals exclude zero in the negative
direction. Adaptation and continuation results are still pending.

# Conclusions
Quality retention requires no measured decrease on any selected benchmark and no held-out NLL increase. Paired uncertainty is reported separately; a single-seed result does not establish robustness. Failure within the pilot budget is not proof that conversion is impossible.

# Changelog
- 2026-10-28: Added the interviewed adaptation design and bounded launch/evaluation paths.

- 2026-10-29: Separated production-dtype same-shape causality checks from the FP64 cross-length reference check; original tolerances retained.

- 2026-10-30: Recorded completed baseline evaluations; untrained conversion loses quality on every selected benchmark.
