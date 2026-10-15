# SPS Experiments

## Language

**Zero-window SPS**:
An SPS variant whose prediction tokens cannot attend to other prediction tokens. A prediction token may attend to itself and to causally visible input tokens.
_Avoid_: No prediction self-attention, zero context

**Pilot comparison**:
A shortened comparison of SPS and a standard Transformer trained on the same data with an equal training-token budget. It tests the direction of an effect without claiming numerical reproduction of the paper.
_Avoid_: Full reproduction
