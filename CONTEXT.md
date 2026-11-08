# SPS Experiments

## Language

**Zero-window SPS**:
An SPS variant whose prediction tokens cannot attend to other prediction tokens. A prediction token may attend to itself and to causally visible input tokens.
_Avoid_: No prediction self-attention, zero context

**Pilot comparison**:
A shortened comparison of SPS and a standard Transformer trained on the same data with an equal training-token budget. It tests the direction of an effect without claiming numerical reproduction of the paper.
_Avoid_: Full reproduction


**SPS speculative decoding**:
Generation in which MTP proposals are verified against the reference SPS model, with an acceptance and correction rule preserving that reference model's output distribution.
_Avoid_: Unverified block generation, guaranteed speedup


**SPS MTP group**:
An ordered set of prediction slots sharing the same visible text prefix and predicting consecutive future tokens. Each slot may attend to earlier prediction slots within its own group, but not to slots in other groups or to future text tokens.
_Avoid_: Ordinary zero-window SPS, teacher-forced future-token chain

**Prediction horizon**:
The offset of a future text token relative to the last visible text token of an MTP group.
_Avoid_: Predict window, generated prefix length


**AR-to-SPS adaptation**:
Conversion of a pretrained autoregressive language model into zero-window SPS, with next-token predictions read from dedicated prediction slots. Quality retention refers to these SPS predictions relative to the original autoregressive model.
_Avoid_: SPS-to-MTP adaptation, unchanged AR quality


**Observed quality retention**:
An adapted model has no decrease in any selected benchmark point estimate and no increase in held-out negative log-likelihood relative to its original model under matched evaluation. This is an empirical pilot outcome, not proof of population-level non-inferiority.
_Avoid_: Proven quality equivalence, statistically guaranteed no degradation


**Matched-verifier speedup**:
The measured speed ratio between speculative decoding and ordinary decoding of the same verifier under a matched workload. Speedups for different verifiers have separate baselines and are accompanied by absolute latency, throughput and quality.
_Avoid_: Shared-baseline speedup across AR and SPS verifiers


**Shared-backbone SPS-MTP**:
An SPS model whose multi-horizon proposal and first-horizon verification use the same backbone weights. Its ordinary-decoding baseline uses that same model with the first prediction slot only.
_Avoid_: Separate draft backbone, unchanged pretrained verifier


**SPS verifier**:
The first-prediction-slot distribution of an SPS model against which speculative proposals are accepted or corrected. In shared-backbone SPS-MTP it is a role of the same model that proposes the block.
_Avoid_: Original AR verifier, external teacher


**AR Qwen control**:
The original instruction-tuned autoregressive Qwen model used as the quality reference and the verifier for the EAGLE-3 comparison arm. It is distinct from the adapted SPS verifier.
_Avoid_: SPS baseline, continued-AR adaptation control


**EAGLE-3 control**:
Speculative decoding of the AR Qwen control with a compatible EAGLE-3 proposer. Its matched-verifier speedup is measured against ordinary decoding of that AR Qwen control.
_Avoid_: EAGLE-3 verification by SPS, common-verifier comparison with SPS-MTP
