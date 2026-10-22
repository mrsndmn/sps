---
status: accepted
date: 2026-10-21
---
# Keep SPS MTP prediction attention local to each prefix group

Extend the pretrained zero-window SPS with an ordered group of future-token
prediction slots per visible prefix. A prediction slot may attend only to earlier
prediction slots in that group and itself, in addition to causally visible real
tokens. Real-token states and the original first prediction remain isolated from
new slots. Unlike widening the existing prediction window, this retains a fixed
reference behavior and prevents other groups or future text from providing
unavailable information. The cost is a larger slot sequence and dedicated mask,
position and inference handling rather than reusing the existing paired layout
unchanged. The actual runtime cost must be measured.

Basis: The user explicitly confirmed the mask and positions in interview round 4
on 2026-10-21, while expanding the experiment to multiple horizon counts.
The paired-layout assumptions are in
`src/modeling/models/reverse_sps/core.py` and
`src/modeling/models/attention/triton_sps_flash_attention.py`.
