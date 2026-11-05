# Hypothesis

The adapted SPS checkpoints retain instruction-following ability relative to the original instruct model and matched AR controls.

# Setup

Evaluation function: `scripts.ifeval_run.evaluate`; registry: the existing instruct `experiment_configs`. The separate launcher reuses the established generation and scoring protocol. Its artifact root is configured by `scripts.ifeval_config`; each arm writes isolated manifests, answer streams and completion markers under `runs/out/<experiment>`.

# Results

Pending. This evaluation runs independently of the full MMLU workers.

# Conclusions

Compare official prompt-level and instruction-level strict/loose metrics separately by generation mode. Preserve the existing checkpoints and MMLU artifacts.

# Changelog

- 2026-11-05: Added independent IFEval workers with durable answer reuse and isolated job submission.

- 2026-11-05: Split thinking and non-thinking into independent workers; resume legacy answers after verifying protocol and checkpoint provenance.
