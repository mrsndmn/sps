"""Term-by-term forward-FLOP accounting for the SPS architecture/size grid.

Shared source of truth so every figure that puts training FLOPs on an axis agrees exactly
(the loss-vs-compute panel and the FLOPs validation-history plot). Component-level accounting
in the DeepMind convention (see A. Casson, "Transformer FLOPs",
adamcasson.com/posts/transformer-flops) -- the full attention-score breakdown (QK-logits +
softmax + attention-value reduction), NOT the 6ND shortcut and NOT the lumped single-term
attention of the Kaplan/OpenAI convention.

Every FLOP is 1 multiply or 1 add, so a matmul with M output elements over a contraction of
length K costs 2 * M * K. The forward pass has three groups, summed over all layers:

  1. PARAMETER MATMULS (2 FLOP / parameter):
       attn_qkv     = 2 * n_layer * d * (3d)      # c_attn  d -> 3d   (Q,K,V)
       attn_project = 2 * n_layer * d * d          # c_proj  d -> d
       ff           = 2 * n_layer * 3 * (d * 3d)   # SwiGLU gate+up+down, each d<->3d
     These sum to 2 * (13 * n_layer * d^2) = 2N, where the 13 = 4 (attention: 3d^2 for QKV
     + 1d^2 for the output projection) + 9 (SwiGLU: 3d^2 each for gate, up, down).

  2. ATTENTION SCORE (does not touch parameters; scales with the attended context, per query):
       qk_logits = 2 * n_layer * n_ctx * d        # Q . K^T
       softmax   = 3 * n_layer * n_head * n_ctx    # ~1.2% of the score work
       reduce    = 2 * n_layer * n_ctx * d        # softmax_weights . V

  3. LM HEAD (2 FLOP / parameter over the vocabulary), charged ONCE:
       logits    = 2 * d * n_vocab

The input embedding is an nn.Embedding lookup (a gather, ~0 FLOP), so vocabulary is charged
only at the head -- we do NOT include the DeepMind one-hot `embeddings` term.

Two multipliers distinguish the predict-token variants (SPS / Delayed State / 2x Memory) from
Standard. They run ONE forward pass over an interleaved length-2T sequence (even slot = the
input/state token, odd slot = the `<predict>` token), so:
  * pass_mult = 2 on the PARAMETER matmuls -- every parameter matmul runs twice per real token.
  * score_mult on the ATTENTION-SCORE terms -- 2 queries per real token, and every query attends
    the full persistent context (~T keys) together with the recent window (~W keys), i.e. ~(T+W)
    keys per query, so over the two queries the factor is 2(T+W)/T = 2 + 2W/T. At W = T every query
    attends the whole 2T sequence, giving exactly 4, which is 2x Memory (SPS with the window opened
    to the whole context).
The LM head is charged once for every architecture (only real tokens predict).

Reference values, for anyone checking a re-derivation: Standard XS forward/token = 1.7388e8;
ratios to Standard at (XS/S/M/L/XL) are 1.72/1.83/1.92/1.96/1.97 for SPS and Delayed State and
2.48/2.55/2.61/2.58/2.52 for 2x Memory. `scripts/figures/plot_loss_vs_compute.py` prints the
per-size table it derives from these functions.
"""
from __future__ import annotations

# Kaplan vocab used for the LM-head term (task-specified; the real padded vocab 50304 is used
# only for parameter sanity checks elsewhere and is not part of the FLOP count).
V_FLOP = 50257
T_CTX = 4096
W_WINDOW = 64
D_HEAD = 64  # head dim is 64 at every size, so n_head = d_model // 64

# scale -> (n_layer, d_model); SwiGLU intermediate is always 3*d_model.
SIZE_DIMS = {
    "xs": (8, 512),
    "s": (12, 768),
    "m": (24, 1024),
    "l": (36, 1280),
    "xl": (48, 1600),
}


def p_nonemb(n_layer: int, d_model: int) -> float:
    """Non-embedding parameters: n_layer * 13 d^2 (4d^2 QKVO + 9d^2 SwiGLU@3d)."""
    return n_layer * 13 * d_model**2


def _forward_flops(n_layer: int, d_model: int, pass_mult: float, score_mult: float) -> float:
    """Forward FLOPs per real token, summing the named components above. `pass_mult` scales the
    parameter matmuls; `score_mult` scales the attention-score terms; the head is charged once."""
    d = d_model
    n_head = d_model // D_HEAD
    # (1) parameter matmuls -- sum to 2N; run pass_mult times per real token
    attn_qkv = pass_mult * (2 * n_layer * d * (3 * d))     # c_attn  d -> 3d
    attn_project = pass_mult * (2 * n_layer * d * d)        # c_proj  d -> d
    ff = pass_mult * (2 * n_layer * 3 * (d * 3 * d))        # SwiGLU gate+up+down @ 3d
    # (2) attention score -- QK-logits + softmax + attention-value reduction, per query
    qk_logits = score_mult * (2 * n_layer * T_CTX * d)     # Q . K^T
    softmax = score_mult * (3 * n_layer * n_head * T_CTX)
    reduce = score_mult * (2 * n_layer * T_CTX * d)        # softmax . V
    # (3) LM head -- charged once (only real tokens predict)
    head = 2 * d * V_FLOP
    return attn_qkv + attn_project + ff + qk_logits + softmax + reduce + head


def forward_flops_per_token(arch: str, n_layer: int, d_model: int) -> float:
    """Forward FLOPs per *real* token, in the DeepMind component convention (see module docstring).

    Standard runs one pass with full single-query attention. The predict-token variants run the
    interleaved length-2T sequence, so their parameter matmuls double (pass_mult=2) and their
    attention score carries 2 queries, each attending the full context (~T keys) plus the window
    (~W keys), giving score_mult = 2 + 2W/T, which reduces to 4 (= 2x Memory) when W = T.
    """
    if arch == "standard":
        return _forward_flops(n_layer, d_model, pass_mult=1, score_mult=1)
    if arch in ("sps", "delayed_state"):
        return _forward_flops(n_layer, d_model, pass_mult=2, score_mult=2 + 2 * W_WINDOW / T_CTX)
    if arch == "two_x_memory":  # SPS with the window opened to the full context (W = T)
        return _forward_flops(n_layer, d_model, pass_mult=2, score_mult=2 + 2 * T_CTX / T_CTX)
    raise ValueError(f"unknown arch {arch!r}")


def training_flops_per_token(arch: str, scale: str) -> float:
    """3x the forward pass (forward + backward) per real token, for the given arch and size."""
    n_layer, d_model = SIZE_DIMS[scale]
    return 3.0 * forward_flops_per_token(arch, n_layer, d_model)
