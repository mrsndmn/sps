# Hypothesis

An earlier checkpoint and a gradual learning-rate schedule reduce loss of instruction-following during AR/SPS adaptation.

# Setup

Training function: `scripts.sched_train.main`; evaluation function: `scripts.sched_eval.evaluate`. The launcher `scripts.sched_run` reuses the original instruct experiment registry, pinned prepared dialogs, model and objective definitions. Variant selection, checkpoint gates, schedule and resource limits are defined in `scripts.sched_config` and `scripts.sched_common`.

Artifacts are isolated under the variant roots selected by `scripts.sched_common.root`. Training writes `runs/out/<experiment>`; each evaluation workload writes `runs/out/<experiment>--<workload>`. Original source artifacts and signatures remain intact.

# Results

Pending.

# Conclusions

Compare the intermediate checkpoints and scheduled reruns against the existing terminal checkpoints. Report instruction-following separately by mode and retain the matched AR controls.

# Changelog

- 2026-11-05: Added isolated intermediate-checkpoint evaluation and warmup/cosine training with resumable optimizer-step scheduling.
