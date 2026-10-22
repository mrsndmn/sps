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
All generation evaluations completed. After-adaptation end-to-end generation
throughput and ratio to the original SPS generator:

| Model | Greedy tokens/s | Greedy ratio | Sampling tokens/s | Sampling ratio |
|---|---:|---:|---:|---:|
| N=2 | 32.31 | 0.358 | 32.33 | 0.367 |
| N=3 | 27.99 | 0.324 | 28.18 | 0.324 |
| N=4 | 24.25 | 0.267 | 24.36 | 0.272 |

Reference throughput ranged from 86.4 to 90.7 tokens/s. Greedy output matched
reference token-for-token on every measured prefix, before and after adaptation;
independent native-reference logits checks had zero error. Sampling ratios use
actual output-token throughput because EOS can produce different output lengths.

For N=2, second-horizon accepted/proposed fraction rose from 2.48% to 21.80% in
greedy and from 16.78% to 24.15% in sampling. After adaptation, third-horizon
fractions were 1.54–1.66% in greedy and 4.86–4.98% in sampling; fourth-horizon
fractions were 0% and 1.02%, respectively. These fractions count all proposals at
a horizon, including those discarded after an earlier rejection; they are not
conditional acceptance among proposals whose earlier slots were all accepted.
Detailed raw counts, timing, baseline and before/after metrics are in report
artifacts. The benchmark uses a small held-out prompt set; results are one-seed
measurements for the configured workload.

## Conclusions
Embedding-only adaptation improves future-token quality and second-horizon
acceptance while preserving first-horizon quality. It does not accelerate the
current decoder: measured throughput is 2.7–3.7 times lower than the native SPS
baseline. N=2 is the strongest first candidate for a subsequent unfreezing
ablation; higher horizons add little accepted output in this experiment.

The native sequential verifier snapshots and replays cache states, and its costs
are included in timing. These results do not determine the potential speed of
parallel verification. Transformer unfreezing and a parallel verifier remain
separate next steps, with no further training initiated automatically.

## Changelog
- 2026-10-21: Completed interactive requirements interview and began implementation.
- 2026-10-22: Verified preparation, reported measured runtime estimates and submitted
  the three adaptation jobs with durable observation.
- 2026-10-22: All adaptations completed; recorded held-out horizon quality and
  submitted greedy/sampling generation evaluations.
- 2026-10-23: Verified all completed generation evaluations; recorded acceptance,
  throughput, exact greedy matching and the sequential-verifier limitation.
