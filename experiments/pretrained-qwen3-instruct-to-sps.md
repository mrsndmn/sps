# Hypothesis

An instruction-tuned Qwen can be transferred to zero-window SPS while retaining instruction following and reasoning quality, using assistant-only adaptation and a frozen source teacher.

# Setup

Training function: `scripts.instruct_train.main`; executable design: `scripts.instruct_config.experiment_configs`. Preparation and bounded workers: `scripts.instruct_run`.

# Results

Pending. Original model, staged SPS CE/KD and continued-AR controls use the isolated instruct suite. Generation, answer validity and length truncation are reported separately from likelihood diagnostics.

# Conclusions

Assess preservation against the original source checkpoint in each generation mode. The CE arm also uses teacher-generated reasoning targets; the KD arm additionally uses online distribution matching. Shared synthetic targets separate these two effects. No conclusion about quality preservation before evaluation completes.

# Changelog

- 2026-11-03: Implemented isolated dialog preparation, assistant-masked training, frozen-teacher distillation and instruct evaluation while retaining the base experiment paths.
