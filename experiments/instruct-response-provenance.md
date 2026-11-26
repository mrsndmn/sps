# Hypothesis
Replacing corpus non-thinking answers with answers from the original instruction model can reduce adaptation-induced instruction loss. Paired prompt selection separates response provenance from changes in prompt composition.

# Setup
`workloads.provenance.provenance_prepare.worker` prepares paired conditions. Preparation settings and allocation are defined in its executable experiment registry. AR/SPS training is a dependent next stage, after accepted-pair coverage and token budgets are inspected.

# Results
Preparation submitted as `lm-mpi-job-3a29891f-4577-41d8-816f-777d136dfa79`; outputs pending.

# Conclusions
Keep thinking answers identical and retain the same accepted prompts in both conditions. Report rejected responses and mode-specific token totals before fixing training budgets. Compare the change in the SPS–AR gap across response sources; do not select answers or checkpoints on the final benchmark.

# Changelog
- 2026-11-10: Independent review approved; submitted preparation and registered durable observation.
- 2026-11-10: Added isolated resumable paired-response preparation using the frozen source model.
