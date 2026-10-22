# Embedding-only MTP adaptation of pretrained SPS

## Hypothesis
A frozen pretrained SPS can support useful multi-token proposals by adapting only
new prediction embeddings, with reference verification preserving generation quality.

## Setup
Training function: `scripts.mtp_train.main`; source registry: `experiment_configs`
in `experiments.py`. Evaluation: `scripts.mtp_evaluate.evaluate`.
Artifacts use a separate suite directory under the existing shared artifact root;
prepared data, source checkpoint, environment and HF cache are reused.

## Results
Preparation completed with 62 correctness tests passing on the worker. Native
reference logits and greedy outputs matched exactly on the source-checkpoint
checks. Measured throughput was approximately 121.7k, 87.3k and 69.5k input tokens/s
for the three horizon counts in registry order. Conservative training-time ranges
are 3.0–4.5, 4.0–5.8 and 4.9–7.0 hours, excluding queue time. All adaptation jobs
completed, taking 2.93, 3.90 and 4.69 hours respectively. Final adaptation checkpoints
and their exact token budgets were verified.

Held-out top-1 accuracy percentages after adaptation:

| Model | Horizon 1 | Horizon 2 | Horizon 3 | Horizon 4 |
|---|---:|---:|---:|---:|
| N=2 | 42.14 | 11.57 | — | — |
| N=3 | 42.15 | 11.38 | 7.90 | — |
| N=4 | 42.15 | 11.19 | 7.88 | 6.82 |

For N=2, second-horizon NLL improved from 7.76871 to 5.90151 and accuracy from
4.12% to 11.57%. Every additional horizon improved over copied-embedding
initialization. Within each run, first-horizon metrics remained exactly unchanged.
The held-out prefix counts and before/after metrics are saved in report artifacts.
All three generation evaluation jobs have been submitted; measured acceptance
and generation-speed results remain pending.

## Conclusions
Embedding-only adaptation improves future-token prediction over initialization,
while preserving first-horizon quality. Acceptance and end-to-end generation
speed are still needed before deciding whether to unfreeze the transformer. The first verified decoder
uses native sequential SPS verification for reference correctness, so its timing
does not establish the potential speed of a future parallel verifier. Positive
speedup is not an assumed outcome.

## Changelog
- 2026-10-21: Completed interactive requirements interview and began implementation.
- 2026-10-22: Verified preparation, reported measured runtime estimates and submitted
  the three adaptation jobs with durable observation.
- 2026-10-22: All adaptations completed; recorded held-out horizon quality and
  submitted greedy/sampling generation evaluations.
