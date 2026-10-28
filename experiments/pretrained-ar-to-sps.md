# Hypothesis
A pretrained autoregressive model can recover its measured benchmark quality after conversion to zero-window SPS. Learning the prediction embedding before unfreezing the backbone may improve recovery relative to immediate full adaptation.

# Setup
Training function: `scripts.ar_sps_train.main`. The `ar-sps` suite selects the executable experiment registry. Source identity, training arguments, job resources, budget and scoring rules are defined in code.

# Results
Not run. The source baseline, untrained conversion, embedding-only intermediate, two-stage adaptation, immediate full adaptation and AR continuation form the comparison.

# Conclusions
Quality retention requires no measured decrease on any selected benchmark and no held-out NLL increase. Paired uncertainty is reported separately; a single-seed result does not establish robustness. Failure within the pilot budget is not proof that conversion is impossible.

# Changelog
- 2026-10-28: Added the interviewed adaptation design and bounded launch/evaluation paths.
