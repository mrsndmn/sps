# Additional speculative proposal baselines

## Hypothesis
History lookup, lightweight target-feature heads and a smaller autoregressive
model can reveal whether SPS-MTP's main limitation is proposal cost, proposal
quality or the absence of sampled-token conditioning within its draft block.

## Setup
Training function: `scripts.spec_baseline_train.main`; evaluation function:
`scripts.spec_baseline_evaluate.evaluate`. The `experiment_configs` registry in
`experiments.py` defines training budgets, models, evaluation horizons and workers.
Existing SPS-MTP continuations are controls in the same evaluation suite.

The heads are a Medusa-style chain adaptation, not a reproduction of Medusa tree
search. They predict later positions from the original target's previous-prefix
features, reused from verification without an extra full Transformer proposal
pass. Every method starts with an original-target anchor. Conditional AR proposal
probabilities are evaluated on the actual earlier sampled draft tokens.

## Results
All training and evaluations completed. Medusa heads trained in 6,636 seconds;
the small AR draft trained in 1,706 seconds. Existing FineWeb memmaps were reused.

History lookup achieved 1.90–2.40× greedy speedup and 0.99–1.00× sampling speedup.
Medusa-style heads achieved 1.14–1.18× greedy and 1.15–1.21× sampling speedup.
Small AR achieved 1.30–1.35× greedy and 0.89–1.09× sampling speedup; its shortest
proposal block was fastest. Full MTP continuations achieved 0.94–0.97× greedy
and 0.83–0.94× sampling speedup.

Detailed acceptance counts, timing breakdowns, paired native rates, input and
checkpoint identities, full job names and source-result hashes are saved in
`run/reports/spec-baselines/results.json` and `generation_summary.csv`.
Greedy outputs matched native completely on 61 or 62 of 64 prompts, depending
on the configuration. Full output equality is therefore not achieved.

## Conclusions
Cheap proposals can outperform more accurate, expensive MTP proposals. History
lookup benefits greatly from repeated greedy continuations, but barely helps
sampling. Small AR has useful sampling acceptance, while serial proposal cost
erases its advantage for longer blocks. Medusa heads have low acceptance but
low proposal cost and show the strongest sampling speedup in this pilot.

These are paired speedups from one timed workload per configuration, after
warmup. Close differences do not establish a stable ranking without repeats.
Absolute native throughput differs across jobs. Block floating point drift
remains a limitation; sampling-law unit tests do not establish bitwise agreement
with the original native generator. The heads are a chain Medusa-style baseline,
not Medusa tree search. The AR model starts from scratch, while the other learned
drafts inherit pretrained SPS representations, so total pretraining budgets differ.

## Changelog
- 2026-10-27: Added history lookup, target-feature heads, small causal AR and
  rerun controls with matched anchors, counters and warmup policy.
- 2026-10-28: Completed all comparisons; recorded acceptance, timing, provenance and numerical limitations.
