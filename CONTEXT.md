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
