#!/usr/bin/env python3
"""Multi-token-ahead linear-probe NLL for the SPS prediction vs. state streams.

For a trained checkpoint, we freeze the backbone and train
small per-offset language-model heads -- one ``Linear(hidden, vocab)`` per look-ahead
``k in 1..K`` per stream, each initialised from the tied output embedding -- then measure
per-``(stream, k)`` validation NLL. The heads are the only trainable parameters.

Streams (post-``output_norm`` hidden states, sliced by parity for the predict-token
families; see ``docs`` below):

* ``full_attention`` (Standard): a single stream ``main`` = ``h`` (B, T, H).
* ``sps`` / ``delayed_state``: ``prediction`` = ``h[:, 1::2]`` (the ``<predict>`` / odd slot,
  which the model natively reads for logits) and ``state`` = ``h[:, 0::2]`` (the input / even
  slot). For ``delayed_state`` the retained state lives in the odd slot and the even/"state"
  slice is the windowed input slot -- interpret/label accordingly.

Alignment (verified in ``reverse_sps/core.py``/``full_attention_model.py``): the hidden at
token ``j`` -- standard ``h_j``, SPS even ``2j``, SPS odd ``2j+1`` -- has seen exactly input
tokens ``u_0..u_j``, so the offset-``k`` target is uniformly ``u_{j+k}`` across all model
types. A target is valid when the source is not eos/pad and no eos falls in ``(j, j+k]`` (no
cross-document prediction); at ``k=1`` this reduces to the native ``(x!=eos)&(x!=pad)`` mask.

Models are read from the native ``.pt`` checkpoints ``scripts/train.py`` writes, through the
same ``analysis_common.load_checkpoint_model`` the sibling analyses use, so ``--spec`` has the
identical ``kind:label:/path/to/checkpoint.pt`` form. Nothing is downloaded.

Run one checkpoint (``--data-root`` is the directory used for data prep, i.e. the corpus is
read from ``<data_root>/data/<dataset>/{train,val}.bin``)::

    uv run python -m scripts.analysis.multitoken_head_nll \\
        --scale s --variant sps-w64 --spec sps:SPS:/path/to/ckpt_final.pt \\
        --data-root <DATA_ROOT> --output-dir <out>

Requires CUDA + Triton for the sps/delayed windowed kernels (standard uses flex/triton).
``scripts/analysis/run_multitoken_head_nll.sh`` submits the full (scale x variant) grid.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make ``modeling`` importable whether or not the venv has an editable install: prefer the
# checkout this file lives in (scripts/analysis/<this> -> repo root -> src).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from modeling.models.model import IGNORE_INDEX, generate_left_padded_document_idx  # noqa: E402

# Shared checkpoint loading / model specs, exactly as the sibling analyses use them.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from scripts.analysis import analysis_common as ga  # noqa: E402

# --- fixed vocabulary / special tokens (GPT-2 padded vocab used throughout the repo) -------
EOS_TOKEN_ID = 50256
PAD_TOKEN_ID = 50303

DEFAULT_DATASET = "fineweb-edu"

# variant (HF repo suffix) -> (model kind, sliding window or None)
VARIANT_SPECS = {
    "full-attention": ("full_attention", None),
    "sps-w64": ("sps", 64),
    "sps-w4096": ("sps", 4096),
    "delayed-state-w64": ("delayed_state", 64),
}

# default per-scale micro-batch (seqlen=1024; the predict-token families double to 2T)
DEFAULT_BATCH_FOR_SCALE = {"xs": 24, "s": 16, "m": 8, "l": 6, "xl": 4}


def streams_for_kind(kind: str) -> list[str]:
    """Ordered stream names to probe for a model kind."""
    if kind == "full_attention":
        return ["main"]
    return ["prediction", "state"]  # odd (native readout) / even (input)


# --------------------------------------------------------------------------------------------
# Checkpoint loading (native .pt, via the shared analysis_common loader)
# --------------------------------------------------------------------------------------------
def load_local_model(spec: "ga.ModelSpec", device: torch.device, *, force_warp_off: bool):
    """Load a locally trained ``.pt`` checkpoint and freeze it for probing.

    Delegates to ``analysis_common.load_checkpoint_model`` -- the same loader the gradient and
    persistent-window analyses use -- so this script reads exactly what ``scripts/train.py``
    writes, with no download and no separate release format. Returns ``(model, metadata)``
    where ``metadata`` is the sanity anchor read off the checkpoint itself.
    """
    loaded = ga.load_checkpoint_model(
        spec, device, warp_specialize=False if force_warp_off else None
    )
    model = loaded.model

    # The tied (wte, lm_head) pair must share one Parameter: the k=1 probe is initialised from
    # wte and asserted against the native head, so a broken tie would silently fail that check.
    model.lm_head.weight = model.transformer.wte.weight
    assert torch.equal(model.lm_head.weight, model.transformer.wte.weight), "weight tie broken"

    emb_std = model.transformer.wte.weight.detach().float().std().item()
    assert emb_std > 0.03, f"embedding std {emb_std:.4f} looks like fresh init -- weights not loaded?"

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # best_val_loss / tokens_seen are what train.py stores; they play the role the released
    # metadata used to (an anchor the sanity ladder prints the native k=1 NLL against).
    checkpoint = torch.load(spec.checkpoint_path, map_location="cpu", weights_only=False)
    metadata = {
        "final_val_loss": checkpoint.get("best_val_loss"),
        "tokens_seen": checkpoint.get("tokens_seen"),
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
    del checkpoint

    print(f"[load] {spec.checkpoint_path}: kind={loaded.kind} emb_std={emb_std:.4f}")
    return model, metadata


# --------------------------------------------------------------------------------------------
# Data: contiguous random windows + document-aware k-ahead targets
# --------------------------------------------------------------------------------------------
def open_bin(path: str) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


def resolve_bin(explicit: str | None, *, data_root: str | None, dataset: str, split: str) -> str:
    """Path to a packed ``{train,val}.bin``.

    ``--train-bin`` / ``--val-bin`` win if given; otherwise resolve
    ``<data_root>/data/<dataset>/<split>.bin`` -- the layout ``src/data/prepare.py`` writes,
    with ``data_root`` the same directory every train/evaluate command takes as
    ``system.data_root``.
    """
    if explicit:
        return explicit
    if not data_root:
        raise SystemExit(
            f"--data-root is required to locate {split}.bin (or pass --{split}-bin explicitly); "
            "use the same directory you ran src/data/prepare.py with"
        )
    path = Path(data_root).expanduser() / "data" / dataset / f"{split}.bin"
    if not path.exists():
        raise SystemExit(f"{path} not found; run src/data/prepare.py system.data_root={data_root}")
    return str(path)


def sample_windows(data: np.memmap, batch_size: int, seqlen: int, k_max: int, rng: np.random.Generator) -> torch.Tensor:
    """Sample ``batch_size`` contiguous windows of length ``seqlen + k_max`` (matches the
    ``estimate_loss`` random-window sampling; ``k_max`` extra tokens give the furthest target)."""
    span = seqlen + k_max
    max_start = len(data) - span - 1
    starts = rng.integers(0, max_start, size=batch_size)
    rows = np.stack([np.asarray(data[s : s + span], dtype=np.int64) for s in starts], axis=0)
    return torch.from_numpy(rows)  # (B, seqlen + k_max), long, on CPU


def build_inputs_and_targets(toks_BxL: torch.Tensor, seqlen: int, k_max: int):
    """Return the model input ``x`` (B, seqlen) and a dict ``k -> masked target`` (B, seqlen).

    A position ``j`` is valid for offset ``k`` when the source token is not eos/pad and no
    document boundary (eos) lies in ``(j, j+k]`` -- i.e. ``doc[j+k] == doc[j]``.
    """
    x_BxT = toks_BxL[:, :seqlen]
    doc = generate_left_padded_document_idx(toks_BxL, eos_token_id=EOS_TOKEN_ID, pad_token_id=PAD_TOKEN_ID)
    base_valid = (x_BxT != EOS_TOKEN_ID) & (x_BxT != PAD_TOKEN_ID)
    doc_src = doc[:, :seqlen]
    targets: dict[int, torch.Tensor] = {}
    for k in range(1, k_max + 1):
        y_k = toks_BxL[:, k : k + seqlen]
        valid = base_valid & (doc[:, k : k + seqlen] == doc_src)
        targets[k] = torch.where(valid, y_k, torch.full_like(y_k, IGNORE_INDEX))
    return x_BxT, targets


# --------------------------------------------------------------------------------------------
# Stream extraction (frozen backbone; no_grad + detach so heads can still backprop)
# --------------------------------------------------------------------------------------------
def extract_streams(model, kind: str, x_BxT: torch.Tensor) -> dict[str, torch.Tensor]:
    """Post-``output_norm`` hidden states per stream, as detached fp32 leaves (B, seqlen, H).

    ``no_grad`` (NOT ``inference_mode``): inference tensors cannot participate in autograd
    even after ``.detach()``, and the heads must backprop through these features.
    """
    with torch.no_grad():
        if kind in ("sps", "delayed_state"):
            _, _, documents_idx_Bx2T = model._expand_real_and_document_idx(x_BxT)
            idx_Bx2T = model.add_predict_tokens(x_BxT)
            h = model.forward_hidden_states(idx_Bx2T, documents_idx_Bx2T=documents_idx_Bx2T)
            streams = {"prediction": h[:, 1::2], "state": h[:, 0::2]}
        else:
            h, _, _ = model.forward_hidden_states(x_BxT, x_BxT.clone())
            streams = {"main": h}
    return {name: t.detach().float() for name, t in streams.items()}


def native_forward_loss(model, x_BxT: torch.Tensor, target_BxT: torch.Tensor):
    """Native model NLL sum/count for a given (input, target) -- the model's own tied head.

    ``target`` is cloned because the standard model masks it in place."""
    with torch.no_grad():
        _, _, stats = model(x_BxT, target_BxT.clone())
    return float(stats["token_nll_sum"]), float(stats["token_nll_count"])


# --------------------------------------------------------------------------------------------
# Probe heads
# --------------------------------------------------------------------------------------------
class ProbeHeads(nn.Module):
    """One ``Linear(hidden, vocab, bias=False)`` per (stream, k), initialised from ``wte``."""

    def __init__(self, hidden: int, vocab: int, streams: list[str], k_max: int, init_weight: torch.Tensor):
        super().__init__()
        self.k_max = k_max
        self.stream_names = list(streams)
        self.by_stream = nn.ModuleDict()
        for name in streams:
            heads = nn.ModuleList()
            for _k in range(1, k_max + 1):
                lin = nn.Linear(hidden, vocab, bias=False)
                with torch.no_grad():
                    lin.weight.copy_(init_weight)  # copy, never alias the frozen tied matrix
                heads.append(lin)
            self.by_stream[name] = heads
        self.to(init_weight.device)

    def head(self, stream: str, k: int) -> nn.Linear:
        return self.by_stream[stream][k - 1]


# --------------------------------------------------------------------------------------------
# Eval
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, kind, heads: ProbeHeads, val_data, *, seqlen, k_max, batch_size, eval_tokens, device, seed):
    """Per-(stream, k) NLL over ``eval_tokens`` from val.bin, keeping per-batch (sum,count)
    for bootstrap CIs. Also returns the native tied-head k=1 NLL on the prediction stream."""
    rng = np.random.default_rng(seed)
    streams = streams_for_kind(kind)
    per_batch: dict[tuple[str, int], list[tuple[float, float]]] = {(s, k): [] for s in streams for k in range(1, k_max + 1)}
    native_sum = native_cnt = 0.0
    tokens_done = 0
    while tokens_done < eval_tokens:
        toks = sample_windows(val_data, batch_size, seqlen, k_max, rng).to(device)
        x_BxT, targets = build_inputs_and_targets(toks, seqlen, k_max)
        feats = extract_streams(model, kind, x_BxT)
        for sname in streams:
            feat = feats[sname]
            V = heads.head(sname, 1).weight.shape[0]
            for k in range(1, k_max + 1):
                logits = heads.head(sname, k)(feat).float()
                tgt = targets[k]
                nll = F.cross_entropy(logits.view(-1, V), tgt.view(-1), ignore_index=IGNORE_INDEX, reduction="sum")
                cnt = (tgt != IGNORE_INDEX).sum()
                per_batch[(sname, k)].append((float(nll), float(cnt)))
        # native tied-head k=1 via the model's own forward (raw next-token target; the model
        # does its own eos/pad masking and, for the standard model, mutates the target in place)
        ns, nc = native_forward_loss(model, x_BxT, toks[:, 1 : 1 + seqlen])
        native_sum += ns
        native_cnt += nc
        tokens_done += batch_size * seqlen
    return per_batch, (native_sum / max(native_cnt, 1.0))


def bootstrap_ci(pairs: list[tuple[float, float]], n_boot: int, rng: np.random.Generator):
    sums = np.array([p[0] for p in pairs], dtype=np.float64)
    counts = np.array([p[1] for p in pairs], dtype=np.float64)
    total_count = counts.sum()
    mean = sums.sum() / max(total_count, 1.0)
    n = len(pairs)
    if n <= 1 or total_count <= 0:
        return mean, mean, mean, float(total_count)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        c = counts[idx].sum()
        boots[b] = sums[idx].sum() / c if c > 0 else np.nan
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi), float(total_count)


# --------------------------------------------------------------------------------------------
# Sanity checks (run before/at start of training)
# --------------------------------------------------------------------------------------------
def run_sanity(model, kind, heads, val_data, metadata, *, seqlen, k_max, batch_size, device, seed):
    """(1) reproduce release final_val_loss via the native head; (2) at head init the
    prediction/main-stream k=1 probe must equal the native head NLL (heads init from wte)."""
    rng = np.random.default_rng(seed)
    # accumulate over a few batches
    probe_sum = probe_cnt = 0.0
    native_sum = native_cnt = 0.0
    n_batches = max(2, min(32, int(1_000_000 / (batch_size * seqlen))))
    read_stream = "prediction" if kind != "full_attention" else "main"
    has_state = "state" in heads.stream_names  # sps/delayed_state carry an even/input slot
    state_k0_sum = state_k0_cnt = 0.0  # init NLL reading the slot's OWN (seen) token u_p
    state_k1_sum = state_k1_cnt = 0.0  # init NLL predicting the NEXT (unseen) token u_{p+1}
    for _ in range(n_batches):
        toks = sample_windows(val_data, batch_size, seqlen, k_max, rng).to(device)
        x_BxT, targets = build_inputs_and_targets(toks, seqlen, k_max)
        feats = extract_streams(model, kind, x_BxT)
        V = heads.head(read_stream, 1).weight.shape[0]
        logits = heads.head(read_stream, 1)(feats[read_stream]).float()
        tgt = targets[1]
        probe_sum += float(F.cross_entropy(logits.view(-1, V), tgt.view(-1), ignore_index=IGNORE_INDEX, reduction="sum"))
        probe_cnt += float((tgt != IGNORE_INDEX).sum())
        ns, nc = native_forward_loss(model, x_BxT, toks[:, 1 : 1 + seqlen])
        native_sum += ns
        native_cnt += nc
        if has_state:
            Vs = heads.head("state", 1).weight.shape[0]
            state_logits = heads.head("state", 1)(feats["state"]).float().view(-1, Vs)
            # offset 0: the state slot's OWN input token u_p -- causally seen, should be easy
            tgt0 = torch.where(
                (x_BxT != EOS_TOKEN_ID) & (x_BxT != PAD_TOKEN_ID),
                x_BxT, torch.full_like(x_BxT, IGNORE_INDEX),
            )
            state_k0_sum += float(F.cross_entropy(state_logits, tgt0.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum"))
            state_k0_cnt += float((tgt0 != IGNORE_INDEX).sum())
            # offset 1: the NEXT token u_{p+1} -- strictly future, must be harder (no leakage)
            state_k1_sum += float(F.cross_entropy(state_logits, targets[1].reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum"))
            state_k1_cnt += float((targets[1] != IGNORE_INDEX).sum())
    probe_k1 = probe_sum / max(probe_cnt, 1.0)
    native_k1 = native_sum / max(native_cnt, 1.0)
    final_val = metadata.get("final_val_loss")
    print(f"[sanity] seqlen={seqlen} native k=1 NLL={native_k1:.4f} "
          f"release final_val_loss={final_val} probe(init) k=1 NLL={probe_k1:.4f}")
    # init-time probe (wte head) must match native head on the same slice, to bf16 noise
    assert abs(probe_k1 - native_k1) < 5e-2, (
        f"init prediction-stream k=1 probe ({probe_k1:.4f}) != native ({native_k1:.4f}); "
        "parity slice / masking / alignment mismatch"
    )
    sanity_out = {"native_k1_nll": native_k1, "probe_init_k1_nll": probe_k1, "release_final_val_loss": final_val}
    if has_state:
        state_k0 = state_k0_sum / max(state_k0_cnt, 1.0)
        state_k1 = state_k1_sum / max(state_k1_cnt, 1.0)
        print(f"[sanity] state-stream init readout (wte head): own-token(offset0) "
              f"NLL={state_k0:.4f}  next-token(offset1) NLL={state_k1:.4f}")
        # Anti-leakage floor. The even/input slot at position p provably sees only u_0..u_p
        # (strict causal mask over the doubled sequence; persistent window off for the released
        # models), so at init -- with the untrained wte-copy head -- it cannot predict the NEXT
        # (unseen) token near-perfectly; a near-zero next-token NLL here would be the fingerprint
        # of embedding-aligned future-token leakage.
        # NB: we deliberately do NOT assert own(offset0) < next(offset1). That holds for SPS,
        # whose even slot is the natively-read persistent state (own~5.8 < next~7.0), but NOT
        # for delayed_state, whose even slot is the windowed *input* slot -- its job is not to be
        # read as logits, so via the tied wte head both offsets sit ABOVE the uniform-vocab level
        # (~10.8 nats) and their ordering is meaningless noise. The decisive no-leakage guard is
        # the post-training floor check in main() (leakage would drive a trained NLL toward 0).
        assert state_k1 > 1.0, (
            f"state-stream next-token NLL at init ({state_k1:.4f}) is near-zero: an UNSEEN token "
            "is predicted almost perfectly from the input slot -- future-token leakage"
        )
        sanity_out["state_init_own_token_nll"] = state_k0
        sanity_out["state_init_next_token_nll"] = state_k1
    return sanity_out


# --------------------------------------------------------------------------------------------
# Main train + eval
# --------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", required=True, choices=["xs", "s", "m", "l", "xl"])
    ap.add_argument("--variant", required=True, choices=list(VARIANT_SPECS))
    ap.add_argument("--spec", required=True,
                    help="model to probe, as kind:label:/path/to/checkpoint.pt "
                         "(same form as the gradient / persistent-window analyses)")
    ap.add_argument("--data-root", default=None,
                    help="dir used for data prep; reads <data-root>/data/<dataset>/{train,val}.bin")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--train-bin", default=None, help="explicit path, overrides --data-root")
    ap.add_argument("--val-bin", default=None, help="explicit path, overrides --data-root")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=None, help="default: per-scale")
    ap.add_argument("--train-tokens", type=float, default=5e7)
    ap.add_argument("--eval-tokens", type=float, default=8e6)
    ap.add_argument("--eval-every-tokens", type=float, default=5e6)
    # 5e-4 is near the floor of the probe's lr curve: an lr sweep at scale s
    # (lr in {3e-4, 5e-4, 1e-3, 2e-3, 3e-3}) showed every k>1 head on both streams is
    # strongly lr-sensitive and badly undertrained at 3e-3 (SPS prediction k=8: 7.72 at
    # 3e-3 vs 7.07 at 3e-4), which even reverses the SPS-vs-Delayed-State input-stream
    # ordering at k=8. The curve flattens below 5e-4 (5e-4 is within 0.015 nats of 3e-4),
    # and 5e-4 converges faster at the large scales inside the fixed --train-tokens budget.
    # k=1 heads are lr-robust either way. Reported figures use this default.
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warp-specialize", choices=["keep", "off"], default="keep")
    ap.add_argument("--sanity-seqlen", type=int, default=None, help="seqlen for the final_val_loss sanity; default = block_size")
    ap.add_argument("--data-seed", type=int, default=1234)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--sanity-only", action="store_true")
    args = ap.parse_args()

    kind, window = VARIANT_SPECS[args.variant]
    spec = ga.parse_model_spec(args.spec)
    # --variant fixes the output layout and the window recorded in the payload; the checkpoint
    # decides what the model actually is. Disagreement means the grid was wired up wrong.
    assert spec.kind == kind, (
        f"--spec kind {spec.kind!r} does not match --variant {args.variant!r} (kind {kind!r})"
    )
    batch_size = args.batch_size or DEFAULT_BATCH_FOR_SCALE[args.scale]
    train_bin = resolve_bin(args.train_bin, data_root=args.data_root,
                            dataset=args.dataset, split="train")
    val_bin = resolve_bin(args.val_bin, data_root=args.data_root,
                          dataset=args.dataset, split="val")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.data_seed)

    print(f"=== multitoken_head_nll: scale={args.scale} variant={args.variant} kind={kind} "
          f"checkpoint={spec.checkpoint_path} device={device} bs={batch_size} "
          f"seqlen={args.seqlen} ===")

    force_warp_off = args.warp_specialize == "off"
    model, metadata = load_local_model(spec, device, force_warp_off=force_warp_off)
    hidden = model.config.hidden_size
    vocab = model.config.vocab_size
    block_size = model.config.block_size
    assert args.seqlen <= block_size, f"seqlen {args.seqlen} > block_size {block_size}"

    streams = streams_for_kind(kind)
    init_weight = model.transformer.wte.weight.detach().clone().float()
    heads = ProbeHeads(hidden, vocab, streams, args.k_max, init_weight).to(device)

    train_data = open_bin(train_bin)
    val_data = open_bin(val_bin)

    # --- sanity ladder (uses the wte-initialised heads before any training) ---
    sanity_seqlen = min(args.sanity_seqlen or block_size, block_size)
    sanity = run_sanity(model, kind, heads, val_data, metadata,
                        seqlen=sanity_seqlen, k_max=args.k_max,
                        batch_size=max(1, batch_size // (sanity_seqlen // args.seqlen or 1)),
                        device=device, seed=args.data_seed + 7)

    out_dir = Path(args.output_dir) / args.scale
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.variant}.json"

    if args.sanity_only:
        print("[sanity-only] stopping after sanity checks.")
        return

    optimizer = torch.optim.AdamW(heads.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    # best (min) val NLL seen per (stream,k) = the early-stopping probe value
    best: dict[tuple[str, int], dict] = {}
    rng = np.random.default_rng(args.data_seed)
    tokens_seen = 0
    step = 0
    t0 = time.time()
    boot_rng = np.random.default_rng(args.data_seed + 99)

    def do_eval(tag):
        # fixed eval seed => every periodic eval scores the SAME val windows, so tracking the
        # per-(stream,k) minimum over the trajectory is a proper early-stopping probe value.
        per_batch, native_k1 = evaluate(model, kind, heads, val_data, seqlen=args.seqlen, k_max=args.k_max,
                                        batch_size=batch_size, eval_tokens=args.eval_tokens, device=device,
                                        seed=args.data_seed + 555)
        for (sname, k), pairs in per_batch.items():
            mean, lo, hi, n_tok = bootstrap_ci(pairs, args.n_boot, boot_rng)
            cur = best.get((sname, k))
            if cur is None or mean < cur["nll"]:
                best[(sname, k)] = {"nll": mean, "ci_lo": lo, "ci_hi": hi, "n_tokens": n_tok, "step": step}
        print(f"[eval {tag}] step={step} tokens={tokens_seen:,} native_k1={native_k1:.4f} "
              + " ".join(f"{s}/k{k}={best[(s,k)]['nll']:.3f}" for s in streams for k in (1, args.k_max)))
        return native_k1

    # step-0 eval: heads are still the wte copy, so this records the native-head baseline
    # as a best-val candidate. At k=1 the prediction/main head starts at the model's own
    # (near-optimal) head; a single lr step overshoots it, so without this the reported k=1
    # probe sits above the native head. init loses to training for k>=2 (wte is far from
    # optimal there), so those curves are unaffected.
    native_k1_final = do_eval(tag="init")
    next_eval = int(args.eval_every_tokens)
    while tokens_seen < args.train_tokens:
        toks = sample_windows(train_data, batch_size, args.seqlen, args.k_max, rng).to(device)
        x_BxT, targets = build_inputs_and_targets(toks, args.seqlen, args.k_max)
        feats = extract_streams(model, kind, x_BxT)
        optimizer.zero_grad(set_to_none=True)
        # per-head backward keeps peak memory at a single (B,seqlen,V) logits tensor
        for sname in streams:
            feat = feats[sname]
            for k in range(1, args.k_max + 1):
                logits = heads.head(sname, k)(feat).float()
                tgt = targets[k]
                cnt = (tgt != IGNORE_INDEX).sum().clamp(min=1)
                loss = F.cross_entropy(logits.view(-1, vocab), tgt.view(-1), ignore_index=IGNORE_INDEX, reduction="sum") / cnt
                loss.backward()
        optimizer.step()
        tokens_seen += batch_size * args.seqlen
        step += 1
        if tokens_seen >= next_eval:
            native_k1_final = do_eval(tag=f"{tokens_seen//1_000_000}M")
            next_eval += int(args.eval_every_tokens)

    if native_k1_final is None or tokens_seen > 0:  # ensure a final eval
        native_k1_final = do_eval(tag="final")

    results = []
    for sname in streams:
        for k in range(1, args.k_max + 1):
            b = best[(sname, k)]
            results.append({
                "scale": args.scale, "variant": args.variant, "kind": kind, "window": window,
                "stream": sname, "k": k, "nll": b["nll"], "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"],
                "n_tokens": b["n_tokens"], "best_step": b["step"],
            })
    payload = {
        "scale": args.scale, "variant": args.variant, "kind": kind, "window": window,
        "checkpoint_path": spec.checkpoint_path, "model_label": spec.label,
        "seqlen": args.seqlen, "k_max": args.k_max, "batch_size": batch_size,
        "train_tokens": tokens_seen, "eval_tokens": args.eval_tokens, "lr": args.lr,
        "hidden_size": hidden, "vocab_size": vocab, "block_size": block_size,
        "native_lmhead_k1_nll": native_k1_final,
        "release_final_val_loss": metadata.get("final_val_loss"),
        "sanity": sanity, "num_parameters": metadata.get("num_parameters"),
        "elapsed_sec": time.time() - t0,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[done] wrote {out_path} in {payload['elapsed_sec']:.0f}s")

    # Post-hoc no-leakage check on the TRAINED state/input stream: genuine k-ahead prediction
    # leaves multi-nat residual NLL, whereas future-token leakage would let a linear probe drive
    # an UNSEEN target's NLL toward zero. Causality is already guaranteed by the strict 2T causal
    # mask (and the init-time causal-wall check in run_sanity); this guards the trained numbers
    # against silently violating it. Placed after the write so results are never lost if it trips.
    state_nlls = [r["nll"] for r in results if r["stream"] == "state"]
    if state_nlls:
        min_state = min(state_nlls)
        print(f"[leak-check] min trained state-stream NLL over k = {min_state:.4f} nats "
              "(floor 0.5; a near-zero value would indicate future-token leakage)")
        assert min_state > 0.5, (
            f"trained state-stream NLL collapsed to {min_state:.4f} nats (< 0.5): a future token "
            "is being read almost perfectly from the input slot, indicating leakage"
        )


if __name__ == "__main__":
    main()
