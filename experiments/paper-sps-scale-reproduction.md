## Hypothesis

Zero-window SPS trained from scratch retains the validation quality reported
for the paper's main SPS models across the selected scales.

## Setup

`training.train.main` uses the reproduction recipes in `experiments.paper`.

## Results

S completed with a final FineWeb-Edu validation NLL of 2.84166. Its final
checkpoint was published. M and XL are still training.

## Conclusions

Compare final FineWeb-Edu validation loss with the published SPS result at
each scale and with the published small-model window ablation.

## Changelog

- 2026-11-26 — Registered the reproduction recipes.
- 2026-11-26 — Added shared corpus preparation and individual MLSpace training commands.
- 2026-11-28 — Recorded the final S validation result.
