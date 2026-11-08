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

The isolated serving runtime passed native CUDA, FlashInfer sampling and
dependency checks. Native vLLM SPS and SPS-MTP greedy outputs exactly matched
the training-runtime reference on the correctness pilot. SPS and EAGLE use
separate matched verifier baselines.

The first completed thinking comparison gives AR and EAGLE-3 the same STEM
accuracy of 62%. Median decode throughput is 43.03 and 68.38 tokens/s respectively;
the median paired throughput speedup is 1.59x. Median request latency is 27.04
and 17.15 seconds. Output limits were reached on 24% and 26% of answers, so this
quality result is limited by the configured generation cap.

SPS thinking achieves 70% accuracy and SPS-MTP thinking 63%. Median decode
throughput is 36.66 and 74.67 tokens/s, with median paired speedup 2.04x
(range 2.00–2.05x across timing repeats). Median request latency is 27.47 and
13.92 seconds. The output cap is reached on 13% and 18% of answers. This
sampled pilot therefore demonstrates acceleration but does not establish
preserved quality. Non-thinking comparisons are pending.

## Conclusions

The thinking pilot shows positive SPS-MTP throughput speedup, together with
lower observed accuracy on the sampled subset. Complete the non-thinking
comparisons before drawing conclusions across generation modes.

## Changelog

- 2026-11-06: Defined shared-backbone Qwen SPS-MTP adaptation and vLLM pilot.
- 2026-11-06: Completed training and recorded held-out loss; serving measurements pending.
- 2026-11-08: Prepared the serving runtime and launched native vLLM greedy verification.
- 2026-11-08: Passed native greedy parity for both SPS arms; started benchmark submission.
- 2026-11-08: Completed the initial AR/EAGLE-3 thinking comparison; SPS evaluation follows.
- 2026-11-08: Completed SPS thinking comparison; measured acceleration and lower sampled accuracy.
