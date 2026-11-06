# Hypothesis
Longer CE adaptation with a learning-rate schedule can recover more of the pretrained model's quality in SPS format.

# Setup
`scripts.base50_train.main`, submitted through `scripts.base50_run.main`. The executable design is in `scripts/base50_config.py`; shared pilot data utilities prepare a separate corpus and artifact root.

# Results
Pending.

# Conclusions
After successful training, remind the user to evaluate cramming on the resulting checkpoint.

# Changelog
- 2026-11-06: Added isolated long base CE training and completion reminder.
