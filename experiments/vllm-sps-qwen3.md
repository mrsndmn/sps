## Hypothesis

A shared SPS backbone with grouped future-token predictions can accelerate
instruction-tuned Qwen in vLLM while preserving the final model's first-slot
sampling distribution.

## Setup

Training function: `scripts.vllm_sps_train.main`. Executable design:
`scripts.vllm_sps_config.experiment_configs`. Evaluation function:
`scripts.vllm_sps_evaluate.evaluate`.

## Results

Training completed and the final shared model was exported to Hugging Face
format. Held-out cross-entropy across the prediction horizons decreased from
2.181 at the first full-backbone validation to 1.603 at the final validation.
The embedding-only stage uses a different loss mask and is not directly comparable.
These losses do not establish first-slot quality, MMLU accuracy or serving speed.

The native vLLM benchmark remains pending: the pinned source requires a runtime
incompatible with the project environment. SPS and EAGLE use separate matched
verifier baselines.

## Conclusions

Measure quality and latency for the resulting shared model. Neither unchanged
parent quality nor positive speedup is assumed.

## Changelog

- 2026-11-06: Defined shared-backbone Qwen SPS-MTP adaptation and vLLM pilot.
- 2026-11-06: Completed training and recorded held-out loss; serving measurements pending.
