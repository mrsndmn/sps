## Hypothesis

A shared SPS backbone with grouped future-token predictions can accelerate
instruction-tuned Qwen in vLLM while preserving the final model's first-slot
sampling distribution.

## Setup

Training function: `scripts.vllm_sps_train.main`. Executable design:
`scripts.vllm_sps_config.experiment_configs`. Evaluation function:
`scripts.vllm_sps_evaluate.evaluate`.

## Results

Pending. Runtime compatibility with fresh vLLM main must be established before
the serving benchmark. SPS and EAGLE use separate matched verifier baselines.

## Conclusions

Measure quality and latency for the resulting shared model. Neither unchanged
parent quality nor positive speedup is assumed.

## Changelog

- 2026-11-06: Defined shared-backbone Qwen SPS-MTP adaptation and vLLM pilot.
