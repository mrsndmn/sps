# Hypothesis
Matching a frozen original AR teacher during both stages of SPS adaptation will
reduce the remaining quality loss compared with the completed plain-CE treatment.

# Setup
Training function: `scripts.ar_sps_train.main`. The distillation suite selects its
registry and objective in executable code. The student starts from the pretrained
source; the completed CE treatment supplies immutable data and evaluation controls.

# Results
Not run. Preparation will validate the reused data and measure training with the
teacher present before permitting the matched-exposure run.

# Conclusions
Compare fixed-exposure endpoints against both original AR and the completed staged
CE model. Validation-selected checkpoints are reported separately. Success retains
the strict no-decrease criterion; improved average quality alone is insufficient.

# Changelog
- 2026-11-01: Added the frozen-source-teacher distillation follow-up.
