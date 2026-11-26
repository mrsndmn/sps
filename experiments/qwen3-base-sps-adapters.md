# Hypothesis
Attention adapters can preserve more source-model quality than full SPS adaptation. Magnitude adaptation and adapter scaling may affect retention.

# Setup
Training function: `workloads.adaptation.qwen_adapter_worker.train`.

# Results
Pending.

# Conclusions
Select the final adapter by validation loss before evaluating language benchmarks.

# Changelog
- 2026-11-21: Added the matched adapter comparison and winner-only benchmark route.
