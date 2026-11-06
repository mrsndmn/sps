# Hypothesis
Additional updates on the fixed instruction corpus can improve instruction following beyond the original final checkpoint.

# Setup
`scripts.extend_train.main`, launched by `scripts.extend_run.main`. The source checkpoints and optimizer states are continued; executable definitions and checkpoint selection live in `scripts/extend_config.py`. IFEval uses the existing protocol in both generation modes.

# Results
Pending.

# Conclusions
Compare retained intermediate checkpoints against the original final checkpoints to separate the effect of more updates from changing the data source.

# Changelog
- 2026-11-06: Added isolated constant-LR continuation with deterministic epoch permutations and milestone evaluations.
