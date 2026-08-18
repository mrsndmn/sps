#!/usr/bin/env python3
"""Pre-decay validation NLL vs. training FLOPs for the Standard/SPS x size grid.

Pulls per-run validation trajectories from W&B (``--entity``/``--project``, defaulting to
``$WANDB_ENTITY`` and the project ``conf/logging/default.yaml`` trains into), truncates to
the constant-LR (pre-decay) phase, converts tokens to
training FLOPs with the term-by-term DeepMind-convention accounting in
``plotting.flops``, Gaussian-smooths each trajectory, and renders a single publication
panel coloured by size / styled by architecture, with a per-architecture compute-optimal
frontier. Also emits a raw-points CSV, a run manifest, a decay-gain table, and a
FLOP-ratio table (which carries the measured GPU-seconds-per-token ratio alongside).

The W&B pull is cached to JSON, so a re-render needs no network::

    uv run python scripts/figures/plot_loss_vs_compute.py            # from the cache
    uv run python scripts/figures/plot_loss_vs_compute.py --force-pull

The first run needs W&B (``--entity``/``--project``); every later render reads the cache and
needs no network at all, so the cache is worth keeping alongside the figure.
"""
from __future__ import annotations

# --- path bootstrap + matplotlib cache pin (before numpy/matplotlib) ---------------
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plotting.mpl_cache  # noqa: F401  (pin MPLCONFIGDIR off NFS before matplotlib)

import argparse
import csv
import json
import math

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "Liberation Serif",
            # STIXGeneral = Times-metric fallback for hosts with no LaTeX and no Times;
            # before DejaVu Serif so it wins there. Ignored under usetex; real Times wins if present.
            "STIXGeneral",
            "DejaVu Serif",
            "serif",
        ],
        # Times-metric-compatible math (matches the paper family) so $10^n$ tick labels render
        # in a serif face beside the Times body text when usetex is off; ignored under usetex.
        "mathtext.fontset": "stix",
    }
)
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, FuncFormatter
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

from plotting.style import (
    _apply_axes_background, LEGEND_TEXT_COLOR, LEGEND_HEADER_COLOR,
    enable_latex_if_available, sc, bf,
)
# Term-by-term forward-FLOP accounting (DeepMind convention) -- the shared source of truth,
# so this panel and the FLOPs history figure put identical numbers on their x-axes.
from plotting.flops import SIZE_DIMS, forward_flops_per_token, p_nonemb
# Shared paper-figure typography (Times sizes/helpers), same source the persistent-window and
# gradient figures use, so this single-panel plot reads as one family with them.
from plotting.figure_common import AXIS_LABEL_FONT_SIZE, _style_ticks

# ============================ constants ============================================
# W&B location comes from --entity/--project (or $WANDB_ENTITY/$WANDB_PROJECT), like every
# other figure script here -- nothing about a particular account is baked in. Only needed
# for a --force-pull; a cached pull renders with no W&B access at all.
DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "sps")
# Cross-check source: the per-step history CSV written by plot_validation_nll_history_wandb.py
# (Fig. 2). Overridable with --crosscheck-csv; skipped with a message when absent.
CSV_CROSSCHECK = _REPO_ROOT / "figures" / "history_combined_gpu.csv"

# The real padded GPT-2 vocab, used only for the parameter sanity check below -- the FLOP
# count charges the Kaplan vocab (plotting.flops.V_FLOP) at the LM head.
V_ACTUAL = 50304

SIZE_ORDER = ["xs", "s", "m", "l", "xl"]
SIZE_LABEL = {"xs": "XS", "s": "S", "m": "M", "l": "L", "xl": "XL"}
# README total parameter counts (for the sanity check).
README_PARAMS = {"xs": 53e6, "s": 131e6, "m": 379e6, "l": 831e6, "xl": 1.678e9}

# architectures plotted/tabulated: Standard vs SPS only (the headline 2-way comparison).
# delayed_state / two_x_memory stay defined in the FLOP formulas but are out of scope here.
ARCH_ORDER = ["standard", "sps"]
ARCH_LABEL = {
    "standard": "Standard",
    "sps": "SPS",
    "delayed_state": "Delayed State",
    "two_x_memory": "2x Memory",
}

# arch -> line style. Standard and SPS are both solid: architecture is already carried by
# HUE (blue vs orange), so a second (line-style) encoding is redundant. delayed_state /
# two_x_memory keep dashes but are out of scope for this 2-way plot.
ARCH_STYLE = {
    "standard": "solid",
    "sps": "solid",
    "delayed_state": (0, (3, 1.4, 1, 1.4)),
    "two_x_memory": (0, (6, 1.4, 1, 1.4, 1, 1.4)),
}

# Architecture is encoded by HUE (Standard=blue, SPS=orange) so the two are instantly
# separable even where curves nearly overlap; size is the light->dark ordinal step within
# each hue. Blue<->orange is the canonical CVD-safe pair; both ramps are monotonic in
# lightness (dataviz "two sequential contexts" pattern). Line style reinforces (solid vs
# dashed) for print / forced-colours.
SIZE_COLOR = {  # Standard: blue ordinal ramp, light->dark = XS->XL
    "xs": "#6da7ec",
    "s": "#3987e5",
    "m": "#256abf",
    "l": "#184f95",
    "xl": "#0d366b",
}
SPS_COLOR = {  # SPS: orange ordinal ramp, light->dark = XS->XL (lightness-matched to blue)
    "xs": "#f0a15e",
    "s": "#e37f34",
    "m": "#cf631c",
    "l": "#a54c14",
    "xl": "#78370e",
}


def curve_color(arch: str, scale: str) -> str:
    return SIZE_COLOR[scale] if arch == "standard" else SPS_COLOR[scale]

# Two compute-optimal frontiers: a saturating power law L(F) = E + A*F^(-alpha) (Chinchilla
# form) fit to each architecture's lower per-size envelope over the displayed FLOP range.
# Loss-vs-log-FLOPs is convex, so a power law hugs the envelope (a straight line floats above
# the curves mid-range and dips below the data at the high-FLOP end). The SPS frontier crosses
# below Standard's near ~1.5e20 FLOPs and keeps pulling ahead -- the headline of the figure.
# Drawn as a thin dotted line (a recessive fitted overlay that must not obscure the real NLL
# curves) over the measured range and faded as it is projected forward past the data, where
# each power law flattens toward its floor E.
FRONTIER_COLOR_BY_ARCH = {"standard": "#0b4f9e", "sps": "#c2410c"}
# Thin dotted line: reads as an estimated trend and, being a fitted overlay, must not obscure
# the real NLL curves it sits over (the data is the primary signal). Kept saturated + on top
# with only a hairline halo so it stays legible over same-hue curves without erasing them.
FRONTIER_STYLE = (0, (1, 2.2))                 # fine dots
FRONTIER_LINEWIDTH = 2.0
FRONTIER_HALO_EXTRA = 1.4                       # halo width = line width + this (hairline ring)
FRONTIER_EXTRAP_MAX = 8e20                     # project the fit this far to show "going forward"

PLOT_DPI = 180  # match the paper family (figure_common.PLOT_DPI)
# Zoom the y-axis to the trained regime so the SPS<->Standard crossover (SPS overtakes
# Standard per-FLOP near the 18B-token budget) is legible; the early high-NLL transient,
# where SPS trails badly per-FLOP, is intentionally cropped (full range is in the CSV).
FLOP_YLIM = (2.42, 3.3)  # bottom lowered from 2.47 to leave room for the forward frontier projection
# Left x-crop at 3e18 FLOPs: below this the curves are still in the steep transient (mostly
# above FLOP_YLIM anyway) and carry no comparison signal, so start the axis there.
FLOP_XMIN = 3e18


# ============================ parameter sanity check ===============================
def implied_total_params(n_layer: int, d_model: int) -> float:
    """13d^2 + tied vocab.d + RMSNorm params (2 per block + final)."""
    norm = (2 * n_layer + 1) * d_model
    return p_nonemb(n_layer, d_model) + V_ACTUAL * d_model + norm


# ============================ W&B run naming =======================================
# arch -> (family, window suffix) in the repo's clean spelling.
ARCH_FAMILY_WINDOW = {
    "standard": ("full_attention", ""),
    "sps": ("sps", "_w64"),
    "delayed_state": ("delayed_state", "_w64"),
    "two_x_memory": ("sps", "_w4096"),
}


def run_name_candidates(arch: str, scale: str, budget: str = "20b") -> list[str]:
    """W&B run names to try for this arch/scale."""
    try:
        family, window = ARCH_FAMILY_WINDOW[arch]
    except KeyError:
        raise ValueError(arch) from None
    return [f"{scale}_{family}{window}_{budget}"]


def run_name(arch: str, scale: str, budget: str = "20b") -> str:
    """The preferred (clean) run name -- used for logging/labelling, not for lookup."""
    return run_name_candidates(arch, scale, budget)[0]


# Cross-check mapping: arch -> (family, window) as spelled in history_combined.csv. That CSV
# carries the *clean* family names (the W&B codenames above are resolved away when it is written),
# so this is deliberately not run_name()'s mapping.
CSV_FAMILY_WINDOW = {
    "standard": ("full_attention", "-1"),
    "sps": ("sps", "64"),
    "delayed_state": ("delayed_state", "64"),
    "two_x_memory": ("sps", "4096"),
}


# ============================ W&B pull (with cache) =================================
def pull_wandb(cache_path: Path, force: bool, *, entity: str | None = None,
               project: str = DEFAULT_PROJECT) -> list[dict]:
    """Return a list of run records; cache to JSON so re-runs skip the network pull."""
    if cache_path.exists() and not force:
        print(f"[cache] loading {cache_path}")
        return json.loads(cache_path.read_text())

    import wandb

    if not entity:
        raise SystemExit(
            "a W&B pull needs an entity: pass --entity <your-entity> or set $WANDB_ENTITY "
            "(no pull is needed if the cache from --cache already exists)"
        )
    wandb_path = f"{entity}/{project}"
    print(f"[pull] {wandb_path}")
    api = wandb.Api(timeout=90)
    records: list[dict] = []

    # (arch, scale, budget) tasks: all four archs at 20b for every size, plus the
    # standard-only 40b constant-LR continuation (18B->36B).
    tasks = [(a, s, "20b") for a in ARCH_ORDER for s in SIZE_ORDER]
    tasks += [("standard", s, "40b") for s in SIZE_ORDER]

    def _dig(d, *path):
        cur = d
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return None
        return cur

    for arch, scale, budget in tasks:
        candidate_names = run_name_candidates(arch, scale, budget)
        name = candidate_names[0]
        cands: list = []
        seen_states: set[str] = set()
        for candidate in candidate_names:
            try:
                found = list(api.runs(wandb_path, filters={"displayName": candidate}, per_page=50))
            except Exception as e:  # noqa: BLE001
                print(f"[warn] query failed for {candidate}: {e!r}")
                continue
            seen_states.update(str(getattr(r, "state", "")) for r in found)
            if any(str(getattr(r, "state", "")).lower() == "finished" for r in found):
                cands, name = found, candidate
                break
            cands = cands or found
        finished = [r for r in cands if str(getattr(r, "state", "")).lower() == "finished"]
        if not finished:
            states = sorted(seen_states) or ["<none>"]
            print(f"[warn] no finished run for {candidate_names} (states={states})")
            continue
        run = max(finished, key=lambda r: str(getattr(r, "created_at", "")))
        cfg = run.config
        rows = [
            (int(r["tokens_seen"]), float(r["val/token_nll"]),
             (float(r["_runtime"]) if r.get("_runtime") is not None else None))
            for r in run.scan_history(
                keys=["_step", "tokens_seen", "val/token_nll", "_runtime"], page_size=5000
            )
            if r.get("tokens_seen") is not None and r.get("val/token_nll") is not None
        ]
        rows.sort(key=lambda t: t[0])
        try:
            gpu_count = int((run.metadata or {}).get("gpu_count") or 0) or None
        except Exception:  # noqa: BLE001
            gpu_count = None
        seed = _dig(cfg, "training", "seed")
        rec = {
            "arch": arch,
            "scale": scale,
            "budget": budget,
            "name": name,
            "run_id": str(run.id),
            "state": str(run.state),
            "created_at": str(getattr(run, "created_at", "")),
            "n_layer": _dig(cfg, "model", "config", "n_layer"),
            "d_model": _dig(cfg, "model", "config", "hidden_size"),
            "window_size": _dig(cfg, "model", "config", "window_size"),
            "vocab_size": _dig(cfg, "model", "config", "vocab_size"),
            "max_tokens": _dig(cfg, "training", "max_tokens"),
            "seed": seed if seed is not None else 1337,
            "gpu_count": gpu_count,
            # final post-decay NLL = last (max-token) history row; the wandb legacy
            # summary API errors on api.runs() objects, and the history tail already
            # contains the post-decay final eval at ~max_tokens.
            "post_decay_nll": (float(rows[-1][1]) if rows else None),
            "history": rows,
        }
        print(f"[pull] {name:26s} id={rec['run_id']:24s} rows={len(rows)} "
              f"gpu={gpu_count} post_nll={rec['post_decay_nll']}")
        records.append(rec)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(records))
    print(f"[cache] wrote {cache_path} ({len(records)} runs)")
    return records


# ============================ trajectory processing ================================
# nominal token budget per label; pre-decay cutoff = 0.9 * nominal (20B->18B, 40B->36B).
NOMINAL_BUDGET_TOKENS = {"20b": 20e9, "40b": 40e9}


def pre_decay_cut_tokens(budget: str) -> float:
    """Pre-decay cutoff = 0.9 * nominal token budget (spec: 20B->18B, 40B->36B).

    Keyed on the nominal budget label rather than config ``max_tokens`` (which is
    absent on the cloned W&B runs); the spec defines the cutoff on nominal budget.
    """
    return 0.9 * NOMINAL_BUDGET_TOKENS[budget]


def gpu_seconds_rate(history: list, gpu_count: int | None) -> float | None:
    """Robust GPU-seconds-per-token rate from interior wall-clock intervals.

    gpu_seconds(token) = token * rate, with rate = median over interior intervals of
    gpu_count * dt_runtime / dt_tokens.  Robust to startup and preemption stalls.
    """
    if not gpu_count:
        return None
    pts = [(t, rt) for (t, _nll, rt) in history if rt is not None and t > 0]
    if len(pts) < 4:
        return None
    pts.sort()
    rates = []
    for (t0, r0), (t1, r1) in zip(pts[1:-1], pts[2:]):  # drop first (warmup) interval
        dt_tok, dt_rt = t1 - t0, r1 - r0
        if dt_tok > 0 and dt_rt > 0:
            rates.append(gpu_count * dt_rt / dt_tok)
    if not rates:
        return None
    return float(np.median(rates))


def smooth_nll(nll: np.ndarray) -> np.ndarray:
    """Smooth a per-token NLL trajectory into a clean publication curve.

    A Gaussian moving average, NOT Savitzky-Golay: a weighted mean cannot introduce a
    maximum or minimum that is not already in the data, so it produces none of the spurious
    "up-and-down" ripples a wide polynomial (poly-3 Savgol) rings off on the steep early
    descent. sigma scales with series length (bounded) so token-space smoothing is roughly
    constant across sizes; mode="nearest" keeps the flat pre-decay tail -- where the
    SPS<->Standard crossover lives -- anchored near its true endpoint.
    """
    n = len(nll)
    if n < 7:
        return nll.copy()
    sigma = float(min(max(4, n // 50), 20))
    return gaussian_filter1d(nll, sigma=sigma, mode="nearest")


def subsample_idx(n: int, target: int = 250) -> np.ndarray:
    if n <= target:
        return np.arange(n)
    step = int(math.ceil(n / target))
    idx = np.arange(0, n, step)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


class Curve:
    """One (arch, size) pre-decay trajectory with FLOPs, GPU-seconds, and smoothing."""

    def __init__(self, arch, scale, tokens, nll, gpu_seconds, seeds, run_names):
        self.arch = arch
        self.scale = scale
        self.n_layer, self.d_model = SIZE_DIMS[scale]
        self.tokens = np.asarray(tokens, float)
        self.nll_raw = np.asarray(nll, float)
        self.gpu_seconds = None if gpu_seconds is None else np.asarray(gpu_seconds, float)
        self.seeds = seeds
        self.run_names = run_names
        self.fwd_per_tok = forward_flops_per_token(arch, self.n_layer, self.d_model)
        self.flops = 3.0 * self.fwd_per_tok * self.tokens
        self.nll_smooth = smooth_nll(self.nll_raw)


def build_curves(records: list[dict]) -> tuple[list[Curve], dict]:
    """Assemble one Curve per (arch, size). Standard concatenates 20b + 40b (18->36B).

    Multi-seed (arch,size,budget) groups are averaged on a shared token grid with a
    min/max band (band info returned alongside; single-seed groups get no band).
    """
    # group finished 20b runs by (arch, scale) [may have >1 seed]; 40b standard separately.
    by_key: dict[tuple, list[dict]] = {}
    std40: dict[str, dict] = {}
    for rec in records:
        if rec["budget"] == "40b" and rec["arch"] == "standard":
            std40[rec["scale"]] = rec
        else:
            by_key.setdefault((rec["arch"], rec["scale"]), []).append(rec)

    curves: list[Curve] = []
    bands: dict[tuple, dict] = {}
    for arch in ARCH_ORDER:
        for scale in SIZE_ORDER:
            recs = by_key.get((arch, scale))
            if not recs:
                continue
            # per-seed pre-decay series
            seed_series = []
            for rec in recs:
                cut = pre_decay_cut_tokens(rec["budget"])
                hist = [(t, nll, rt) for (t, nll, rt) in rec["history"] if 0 < t <= cut]
                if not hist:
                    continue
                # dedupe by token (keep last), sort
                dd = {}
                for t, nll, rt in hist:
                    dd[t] = (nll, rt)
                toks = sorted(dd)
                nll = [dd[t][0] for t in toks]
                rt = [dd[t][1] for t in toks]
                rate = gpu_seconds_rate(rec["history"], rec["gpu_count"])
                gsec = ([rate * t for t in toks] if rate else None)
                seed_series.append(dict(rec=rec, tokens=toks, nll=nll, gsec=gsec, rate=rate))
            if not seed_series:
                continue

            # standard: append the 40b continuation (18->36B) to each seed series' base run
            if arch == "standard" and scale in std40:
                rec40 = std40[scale]
                cut40 = pre_decay_cut_tokens(rec40["budget"])  # 36e9
                base_cut = pre_decay_cut_tokens(recs[0]["budget"])  # 18e9
                ext = [(t, nll, rt) for (t, nll, rt) in rec40["history"]
                       if base_cut < t <= cut40]
                dd = {}
                for t, nll, rt in ext:
                    dd[t] = (nll, rt)
                etoks = sorted(dd)
                if etoks:
                    rate40 = gpu_seconds_rate(rec40["history"], rec40["gpu_count"])
                    ss = seed_series[0]
                    seam = dd[etoks[0]][0] - ss["nll"][-1]
                    print(f"[seam] standard/{scale}: NLL gap at 18B "
                          f"(40b_first {dd[etoks[0]][0]:.4f} - 20b_last "
                          f"{ss['nll'][-1]:.4f}) = {seam:+.4f}")
                    # gpu-seconds of the extension continues from the base run's end
                    if ss["gsec"] is not None and rate40:
                        base_end_g = ss["gsec"][-1]
                        base_end_t = ss["tokens"][-1]
                        gext = [base_end_g + rate40 * (t - base_end_t) for t in etoks]
                        ss["gsec"] = ss["gsec"] + gext
                    elif ss["gsec"] is not None:
                        ss["gsec"] = ss["gsec"] + [ss["gsec"][-1]] * len(etoks)
                    ss["tokens"] = ss["tokens"] + list(etoks)
                    ss["nll"] = ss["nll"] + [dd[t][0] for t in etoks]
                    ss["run_names"] = [ss["rec"]["name"], rec40["name"]]

            seeds = sorted({s["rec"]["seed"] for s in seed_series})
            run_names = sorted({n for s in seed_series for n in
                                (s.get("run_names") or [s["rec"]["name"]])})

            if len(seed_series) == 1:
                ss = seed_series[0]
                curves.append(Curve(arch, scale, ss["tokens"], ss["nll"], ss["gsec"],
                                     seeds, run_names))
            else:
                # interpolate each seed onto the shared (overlapping) token grid, mean+band
                lo = max(min(s["tokens"]) for s in seed_series)
                hi = min(max(s["tokens"]) for s in seed_series)
                grid = np.linspace(lo, hi, 400)
                stack = np.vstack([np.interp(grid, s["tokens"], s["nll"]) for s in seed_series])
                mean = stack.mean(axis=0)
                gsec = None
                if all(s["gsec"] is not None for s in seed_series):
                    gstack = np.vstack([np.interp(grid, s["tokens"], s["gsec"])
                                        for s in seed_series])
                    gsec = gstack.mean(axis=0)
                c = Curve(arch, scale, grid, mean, gsec, seeds, run_names)
                bands[(arch, scale)] = dict(tokens=grid, lo=stack.min(axis=0),
                                            hi=stack.max(axis=0), flops=c.flops)
                curves.append(c)
    return curves, bands


# ============================ compute-optimal frontier =============================
def _fit_powerlaw(lgv: np.ndarray, envv: np.ndarray):
    """Fit the envelope to L = E + A*exp(-beta*(x-x0)) with x = log10(FLOPs).

    A power law in FLOPs, L = E + A'*F^(-alpha), is exactly an exponential decay in log10(F),
    so we fit it in the (recentred) log-FLOP coordinate for numerical conditioning. Returns
    (predict_fn, E, A, alpha) or None if the nonlinear solve does not converge.
    """
    x0 = float(lgv.min())
    xc = lgv - x0                       # recentre so exp(-beta*x) is O(1) across the data
    span = float(xc.max()) or 1.0
    y_lo_obs = float(envv.min())        # best (lowest) NLL -> upper bound on the floor E
    y_hi_obs = float(envv.max())        # worst NLL, at the low-FLOP end (x ~ x0)

    def model(x, E, A, beta):
        return E + A * np.exp(-beta * x)

    # initial guess: floor just under the best point; amplitude = NLL drop across the range;
    # beta from the two endpoints (y_hi-E)=A and (y_lo-E)=A*exp(-beta*span).
    E0 = max(0.0, y_lo_obs - 0.15)
    A0 = max(1e-3, y_hi_obs - E0)
    ratio = (y_lo_obs - E0) / (y_hi_obs - E0) if y_hi_obs > E0 else 0.5
    beta0 = max(1e-2, -math.log(min(max(ratio, 1e-3), 0.999)) / span)
    try:
        popt, _ = curve_fit(
            model, xc, envv, p0=[E0, A0, beta0],
            bounds=([0.0, 1e-4, 1e-3], [y_lo_obs, 10.0, 20.0]),
            maxfev=20000,
        )
    except Exception:  # noqa: BLE001  RuntimeError on non-convergence, ValueError on bad bounds
        return None
    E, A, beta = (float(v) for v in popt)
    alpha = beta / math.log(10.0)       # exponent on F in L = E + A*F^(-alpha) after recentring

    def predict(F):
        x = np.log10(np.asarray(F, dtype=float)) - x0
        return E + A * np.exp(-beta * x)

    return predict, E, A, alpha


def method_frontier(curves: list[Curve], arch: str):
    """Saturating power-law fit of one architecture's compute-optimal frontier.

    The frontier is the lower NLL envelope across the architecture's per-size curves (min over
    whichever sizes reach each FLOP budget). Loss-vs-log-FLOPs is convex (diminishing returns),
    so we fit the Chinchilla-style power law L(F) = E + A*F^(-alpha) -- equivalently exponential
    decay in log10(F) -- which hugs the convex envelope and, projected forward, flattens toward
    the floor E rather than diverging. Returns a dict with a vectorised ``predict(F)`` callable
    plus the fitted params, the fit range [lo, hi], and the envelope RMSE; None if fewer than
    two sizes are present. Falls back to the old log-linear fit if the nonlinear solve fails.
    """
    sel = [c for c in curves if c.arch == arch]
    if len(sel) < 2:
        return None
    log_lo = max(math.log10(FLOP_XMIN), min(np.log10(c.flops.min()) for c in sel))
    log_hi = max(np.log10(c.flops.max()) for c in sel)
    grid = np.logspace(log_lo, log_hi, 400)
    lg = np.log10(grid)
    # per-curve NLL interpolated in log-FLOP space, NaN outside each curve's own range
    mats = []
    for c in sel:
        clg = np.log10(c.flops)
        order = np.argsort(clg)
        mats.append(np.interp(lg, clg[order], c.nll_smooth[order], left=np.nan, right=np.nan))
    M = np.vstack(mats)
    coverage = np.sum(~np.isnan(M), axis=0)
    env = np.nanmin(np.where(np.isnan(M), np.inf, M), axis=0)
    # keep the envelope wherever any size reaches that budget (at the high-FLOP end this is the
    # single largest model -- exactly the regime where SPS pulls ahead, so we must not drop it)
    valid = coverage >= 1
    lgv, envv = lg[valid], env[valid]

    # env_F/env_nll = the empirical lower envelope itself (min NLL over sizes at each FLOP grid
    # point). The power-law path fits it; the "envelope" frontier mode draws it directly.
    base = {"lo": float(10.0 ** log_lo), "hi": float(10.0 ** log_hi),
            "env_F": 10.0 ** lgv, "env_nll": envv}
    fit = _fit_powerlaw(lgv, envv)
    if fit is not None:
        predict, E, A, alpha = fit
        rmse = float(np.sqrt(np.mean((envv - predict(10.0 ** lgv)) ** 2)))
        return {**base, "predict": predict, "kind": "powerlaw",
                "E": E, "A": A, "alpha": alpha, "rmse": rmse}
    # fallback: original straight-line fit (kept so the figure still renders if curve_fit fails)
    slope, intercept = (float(v) for v in np.polyfit(lgv, envv, 1))
    rmse = float(np.sqrt(np.mean((envv - (slope * lgv + intercept)) ** 2)))

    def predict(F):
        return slope * np.log10(np.asarray(F, dtype=float)) + intercept

    return {**base, "predict": predict, "kind": "loglinear",
            "slope": slope, "intercept": intercept, "rmse": rmse}


# ============================ plotting =============================================
def _fmt_pow10(v, _pos):
    if v <= 0:
        return ""
    e = int(round(math.log10(v)))
    return rf"$10^{{{e}}}$"


def _draw_compute_legend(legend_ax, sizes_present, arch_present, frontier_archs,
                         frontier_mode="powerlaw"):
    """Band-style legend for the size x architecture (lightness x hue) colour encoding.

    Left block "Model size": a 2-row colour matrix -- a Standard (blue) ramp row above an SPS
    (orange) ramp row, sharing XS..XL size columns -- so every real curve colour appears and
    the hue<->architecture / lightness<->size mapping reads directly (no hue-neutral proxy).
    Right block "Compute-optimal frontier": the dashed fitted-line swatches on matching
    Standard/SPS rows. Positioned below the plot, in the house style (semibold muted-ink titles,
    frameless) so it sits consistently with the combined history figures' legend band.
    """
    legend_ax.set_axis_off()
    tf = legend_ax.transAxes
    head = dict(color=LEGEND_HEADER_COLOR, fontweight="semibold", fontsize=16, va="center")
    txt = dict(color=LEGEND_TEXT_COLOR, fontsize=14, va="center")
    hdr = dict(color=LEGEND_TEXT_COLOR, fontsize=13, va="center", ha="center")
    ramp = {"standard": SIZE_COLOR, "sps": SPS_COLOR}
    y_title, y_hdr = 0.90, 0.60
    y_row = {"standard": 0.36, "sps": 0.10}

    # ---- left block: Model-size colour matrix ----
    legend_ax.text(0.015, y_title, bf("Model size"), ha="left", transform=tf, **head)
    x0, dx, chip = 0.145, 0.070, 0.045
    col_x = [x0 + i * dx for i in range(len(sizes_present))]
    for cx, s in zip(col_x, sizes_present, strict=True):
        legend_ax.text(cx + chip / 2, y_hdr, SIZE_LABEL[s], transform=tf, **hdr)
    for a in arch_present:
        y = y_row[a]
        legend_ax.text(0.015, y, sc(ARCH_LABEL[a]), ha="left", transform=tf, **txt)
        for cx, s in zip(col_x, sizes_present, strict=True):
            legend_ax.plot([cx, cx + chip], [y, y], color=ramp[a][s], lw=4.0,
                           solid_capstyle="round", transform=tf, clip_on=False)

    # ---- right block: compute-optimal frontier (rows aligned with the matrix) ----
    if frontier_archs:
        # Both modes estimate the same object -- the compute-optimal frontier -- so both use
        # the paper's term; they differ only in whether it is the empirical min over sizes or
        # a power law fitted to it. ("Frontier: min over sizes" was only needed to tell the
        # two apart back when both variants were written side by side.)
        _ftitle = "Compute-optimal frontier"
        legend_ax.text(0.655, y_title, bf(_ftitle), ha="left", transform=tf, **head)
        fx, fchip = 0.665, 0.075
        for a in frontier_archs:
            y = y_row[a]
            legend_ax.plot([fx, fx + fchip], [y, y], color=FRONTIER_COLOR_BY_ARCH[a],
                           lw=FRONTIER_LINEWIDTH, linestyle=FRONTIER_STYLE, dash_capstyle="round",
                           transform=tf, clip_on=False)
            legend_ax.text(fx + fchip + 0.02, y, sc(ARCH_LABEL[a]), ha="left", transform=tf, **txt)


def make_figure(curves, bands, frontiers, x_key, out_png, out_pdf,
                frontier_mode="powerlaw"):
    # Plot on top, a horizontal legend band below -- same organisation as the combined history
    # figures (grouped, semibold muted-ink titles, frameless; via _render_m_classic_legend).
    fig = plt.figure(figsize=(9.8, 8.0))
    grid = fig.add_gridspec(2, 1, height_ratios=[6.2, 1.6], hspace=0.28)
    ax = fig.add_subplot(grid[0, 0])
    legend_ax = fig.add_subplot(grid[1, 0])

    def xvals(c):
        return c.flops if x_key == "flops" else c.gpu_seconds

    plotted = [c for c in curves if xvals(c) is not None]
    if not plotted:
        print(f"[skip] no curves have {x_key}; not rendering {out_png.name}")
        plt.close(fig)
        return False

    all_x = np.concatenate([xvals(c)[xvals(c) > 0] for c in plotted])
    x_hi = all_x.max()
    x_left = FLOP_XMIN if x_key == "flops" else all_x.min()  # crop the empty early region
    # On the FLOPs axis with the fitted power law, extend the right edge past the data so the
    # frontiers can be seen projecting forward ("going forward"); the empirical envelope has no
    # projection, so pad to the data as on the other axes.
    if x_key == "flops" and frontier_mode == "powerlaw":
        x_right = max(x_hi * 1.35, FRONTIER_EXTRAP_MAX * 1.06)
    else:
        x_right = x_hi * 1.35
    y_lo, y_hi = FLOP_YLIM  # zoom to the trained/crossover band (see FLOP_YLIM note)
    ax.set_xscale("log")
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_lo, y_hi)
    _apply_axes_background(ax, x_min=x_left, x_max=x_right, y_min=y_lo, y_max=y_hi)

    # compute-optimal frontiers (bold, above the curves) — only for the FLOPs axis. One line per
    # architecture. In "powerlaw" mode: the solid-weight dashed fit over the measured range,
    # faded as it is projected past the data. In "envelope" mode: the empirical lower envelope
    # itself (min NLL over sizes at each FLOP budget) drawn directly, with no forward projection
    # (an envelope has no meaning past the data). Legend swatches are built for the band below.
    if x_key == "flops":
        for a in ARCH_ORDER:
            fr = frontiers.get(a)
            if fr is None:
                continue
            col = FRONTIER_COLOR_BY_ARCH[a]
            halo_lw = FRONTIER_LINEWIDTH + FRONTIER_HALO_EXTRA
            if frontier_mode == "envelope":
                # empirical min-over-sizes envelope, drawn as-is
                fx, fy = np.asarray(fr["env_F"]), np.asarray(fr["env_nll"])
            else:
                # fitted frontier over the measured FLOP range
                fx = np.logspace(np.log10(fr["lo"]), np.log10(fr["hi"]), 200)
                fy = fr["predict"](fx)
            fline, = ax.plot(fx, fy, color=col,
                             linestyle=FRONTIER_STYLE, linewidth=FRONTIER_LINEWIDTH, zorder=6,
                             solid_capstyle="round", dash_capstyle="round")
            fline.set_path_effects([
                path_effects.Stroke(linewidth=halo_lw, foreground="white", alpha=0.85),
                path_effects.Normal(),
            ])
            # forward projection (faded), truncated at the y-floor so it never runs off-frame --
            # power-law frontier only; the empirical envelope stops at the last measured point.
            if frontier_mode == "powerlaw" and FRONTIER_EXTRAP_MAX > fr["hi"]:
                ex = np.logspace(np.log10(fr["hi"]), np.log10(FRONTIER_EXTRAP_MAX), 200)
                ey = fr["predict"](ex)
                keep = ey >= y_lo
                if keep.any():
                    exline, = ax.plot(ex[keep], ey[keep], color=col, linestyle=FRONTIER_STYLE,
                                      linewidth=FRONTIER_LINEWIDTH, alpha=0.5, zorder=6,
                                      solid_capstyle="round", dash_capstyle="round")
                    exline.set_path_effects([
                        path_effects.Stroke(linewidth=halo_lw, foreground="white", alpha=0.45),
                        path_effects.Normal(),
                    ])

    # bands (seed spread)
    for (arch, scale), b in bands.items():
        bx = b["flops"] if x_key == "flops" else None
        if bx is None:
            continue
        ax.fill_between(bx, b["lo"], b["hi"], color=curve_color(arch, scale), alpha=0.15,
                        linewidth=0, zorder=1.5)

    # curves: smoothed only. The raw per-token trace is intentionally NOT drawn -- it read as
    # noise; the full raw series is still preserved in loss_vs_compute_points.csv.
    for c in plotted:
        x = xvals(c)
        col = curve_color(c.arch, c.scale)
        ls = ARCH_STYLE[c.arch]
        si = subsample_idx(len(x))
        ax.plot(x[si], c.nll_smooth[si], color=col, alpha=0.98, linewidth=2.3,
                linestyle=ls, zorder=4, solid_capstyle="round", dash_capstyle="round")

    # Reference-budget markers are deliberately absent: the two Standard dots (18B and 36B) read
    # as ambiguous, and each curve's endpoint already shows where its run stops. tokens-per-param
    # @18B is a per-size scalar (identical for both archs) and is carried in the size-legend
    # labels rather than inline, to keep the crossover band uncluttered.

    # axes cosmetics
    ax.set_xlabel(bf("Training FLOPs" if x_key == "flops" else "Cumulative GPU-seconds"),
                  fontsize=AXIS_LABEL_FONT_SIZE, fontweight="bold", color=LEGEND_TEXT_COLOR)
    ax.set_ylabel(bf("Pre-decay validation NLL"), fontsize=AXIS_LABEL_FONT_SIZE,
                  fontweight="bold", color=LEGEND_TEXT_COLOR)
    if x_key == "flops":
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_pow10))
    _style_ticks(ax)

    # ---- legend band below the plot: a size x architecture colour matrix (see _draw_compute
    # _legend). Blue ramp = Standard, orange ramp = SPS, light->dark = XS->XL; dashed swatches =
    # the fitted frontiers. Every swatch is a real colour/style from the plot.
    sizes_present = [s for s in SIZE_ORDER if any(c.scale == s for c in plotted)]
    arch_present = [a for a in ARCH_ORDER if any(c.arch == a for c in plotted)]
    frontier_archs = ([a for a in arch_present if frontiers.get(a) is not None]
                      if x_key == "flops" else [])
    _draw_compute_legend(legend_ax, sizes_present, arch_present, frontier_archs,
                         frontier_mode=frontier_mode)

    # ---- programmatic clipping check (final data point inside the axes bbox) ----
    fig.canvas.draw()
    clipped = _check_clipping(ax, plotted, xvals)
    if clipped:
        print(f"[clip] WARNING: {len(clipped)} artist(s) clipped in {out_png.name}: {clipped}")
    else:
        print(f"[clip] OK: nothing clipped in {out_png.name}")

    fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[save] {out_png}\n[save] {out_pdf}")
    return True


def _check_clipping(ax, plotted, xvals):
    """Return a description of every curve whose final point falls outside the axes."""
    problems = []
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    for c in plotted:
        x = xvals(c)
        if x is None:
            continue
        # both axes are intentionally cropped to the trained/crossover region, so early
        # points fall outside by design; require each curve's FINAL point to be visible.
        fx, fy = float(x[-1]), float(c.nll_smooth[-1])
        if not (min(x0, x1) <= fx <= max(x0, x1)):
            problems.append(f"{c.arch}/{c.scale} final x {fx:.2e} outside range")
        if not (min(y0, y1) <= fy <= max(y0, y1)):
            problems.append(f"{c.arch}/{c.scale} final NLL {fy:.3f} outside band")
    return problems


# ============================ CSV cross-check ======================================
def csv_crosscheck(curves, csv_path: Path = CSV_CROSSCHECK):
    """Compare each 20b (arch,size) final pre-decay NLL vs history_combined.csv."""
    if not csv_path.exists():
        print(f"[xcheck] {csv_path} not found; skipping (render the history figure with "
              "--x-axis combined to produce it)")
        return
    # last validation_nll per (scale, family, window)
    last = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            key = (row["scale"], row["family"], row["window"])
            t = float(row["tokens_seen"])
            prev = last.get(key)
            if prev is None or t > prev[0]:
                last[key] = (t, float(row["validation_nll"]))
    print("[xcheck] final pre-decay NLL W&B vs CSV (xs/s/m/l):")
    for c in curves:
        fam, win = CSV_FAMILY_WINDOW[c.arch]
        key = (c.scale, fam, win)
        if key not in last:
            continue
        wb = c.nll_raw[-1] if c.arch != "standard" else c.nll_raw[
            int(np.argmin(np.abs(c.tokens - 18e9)))]
        csv_t, csv_nll = last[key]
        d = wb - csv_nll
        flag = "" if abs(d) < 0.05 else "  <-- CHECK"
        print(f"    {c.arch:14s} {c.scale:3s}  W&B={wb:.4f}  CSV={csv_nll:.4f}  d={d:+.4f}{flag}")


# ============================ outputs ==============================================
def write_points_csv(curves, path):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["architecture", "size", "run_names", "seeds", "n_layer", "d_model",
                    "tokens", "pre_decay_nll_raw", "pre_decay_nll_smoothed",
                    "forward_flops_per_token", "training_flops", "gpu_seconds"])
        for c in curves:
            for i in range(len(c.tokens)):
                w.writerow([
                    c.arch, c.scale, "|".join(c.run_names), "|".join(map(str, c.seeds)),
                    c.n_layer, c.d_model, int(c.tokens[i]),
                    f"{c.nll_raw[i]:.6f}", f"{c.nll_smooth[i]:.6f}",
                    f"{c.fwd_per_tok:.6e}", f"{c.flops[i]:.6e}",
                    ("" if c.gpu_seconds is None else f"{c.gpu_seconds[i]:.3f}"),
                ])
    print(f"[save] {path}")


def write_manifest(records, curves, path):
    present = {(r["arch"], r["scale"]) for r in records if r["budget"] == "20b"}
    lines = ["RUN MANIFEST — loss vs compute", "=" * 60, ""]
    lines.append(f"{'run_name':28s} {'arch':13s} {'size':4s} {'seed':5s} "
                 f"{'run_id':26s} {'n':5s} {'tok_min':>10s} {'tok_max':>12s}")
    for rec in sorted(records, key=lambda r: (r["arch"], SIZE_ORDER.index(r["scale"]), r["budget"])):
        hist = [(t, n) for (t, n, _rt) in rec["history"] if t > 0]
        tmin = min(t for t, _ in hist) if hist else 0
        tmax = max(t for t, _ in hist) if hist else 0
        lines.append(f"{rec['name']:28s} {rec['arch']:13s} {rec['scale']:4s} "
                     f"{str(rec['seed']):5s} {rec['run_id']:26s} {len(hist):5d} "
                     f"{tmin:10d} {tmax:12d}")
    lines += ["", f"Missing (architecture, size) combos among the "
              f"{len(ARCH_ORDER)}x{len(SIZE_ORDER)} grid @20b (scope: Standard vs SPS):"]
    missing = [(a, s) for a in ARCH_ORDER for s in SIZE_ORDER if (a, s) not in present]
    lines.append("  NONE — full grid present" if not missing
                 else "  " + ", ".join(f"{a}/{s}" for a, s in missing))
    lines += ["", "Seeds: main 20b grid is single-seed (seed 1337). Multi-seed studies "
              "exist only for S@10b (different budget) and are NOT merged here."]
    path.write_text("\n".join(lines) + "\n")
    print(f"[save] {path}")
    print("\n".join(lines))


def write_decay_gain(records, curves, path):
    post = {(r["arch"], r["scale"]): r["post_decay_nll"]
            for r in records if r["budget"] == "20b"}
    rows = []
    for c in curves:
        # final pre-decay at the 18B (nominal 20b) cutoff
        j = int(np.argmin(np.abs(c.tokens - 18e9)))
        pre = float(c.nll_raw[j])
        pd = post.get((c.arch, c.scale))
        gain = (pre - pd) if pd is not None else None
        rows.append((c.arch, c.scale, pre, pd, gain))
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["architecture", "size", "final_pre_decay_nll",
                    "final_post_decay_nll", "decay_gain"])
        for a, s, pre, pd, g in rows:
            w.writerow([a, s, f"{pre:.6f}", ("" if pd is None else f"{pd:.6f}"),
                        ("" if g is None else f"{g:.6f}")])
    gains = [g for *_r, g in rows if g is not None]
    print(f"[save] {path}")
    if gains:
        arr = np.array(gains)
        print(f"[decay-gain] n={len(arr)} mean={arr.mean():.4f} std={arr.std():.4f} "
              f"min={arr.min():.4f} max={arr.max():.4f}")
        # per-arch mean (the invariance question is about ARCH, holding size aside)
        arch_means = {}
        for a in ARCH_ORDER:
            ag = [g for (aa, _s, _p, _pd, g) in rows if aa == a and g is not None]
            if ag:
                arch_means[a] = float(np.mean(ag))
                print(f"    {a:14s} mean gain={arch_means[a]:.4f}  (n={len(ag)})")
        overall_spread = float(arr.max() - arr.min())
        arch_spread = (max(arch_means.values()) - min(arch_means.values())
                       if len(arch_means) > 1 else 0.0)
        print(f"[decay-gain] per-arch mean spread={arch_spread:.4f} "
              f"(overall point spread={overall_spread:.4f} is dominated by SIZE, not arch) -> "
              + ("decay gain is approximately ARCHITECTURE-INVARIANT: pre-decay NLL gaps "
                 "carry over to post-decay" if arch_spread < 0.02
                 else "MATERIAL architecture differences in decay gain: pre-decay gaps may "
                 "NOT transfer directly to post-decay"))
    return rows


def write_flop_ratios(curves, records, path):
    # measured gpu-seconds-per-token rate per (arch,size) from records
    rate = {}
    for rec in records:
        if rec["budget"] != "20b":
            continue
        r = gpu_seconds_rate(rec["history"], rec["gpu_count"])
        if r:
            rate[(rec["arch"], rec["scale"])] = r
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size", "n_layer", "d_model"]
                   + [f"fwd_flops_tok_{a}" for a in ARCH_ORDER]
                   + ["implied_params", "readme_params", "param_err_frac",
                      "sps_over_standard_flops", "sps_over_standard_measured_walltime"])
        print("[flop-ratios] per-size forward FLOPs/token and SPS/standard ratio:")
        for s in SIZE_ORDER:
            nl, d = SIZE_DIMS[s]
            ff = {a: forward_flops_per_token(a, nl, d) for a in ARCH_ORDER}
            imp = implied_total_params(nl, d)
            err = abs(imp - README_PARAMS[s]) / README_PARAMS[s]
            ratio = ff["sps"] / ff["standard"]
            meas = ""
            if ("sps", s) in rate and ("standard", s) in rate:
                meas = rate[("sps", s)] / rate[("standard", s)]
            w.writerow([s, nl, d]
                       + [f"{ff[a]:.6e}" for a in ARCH_ORDER]
                       + [f"{imp:.6e}", f"{README_PARAMS[s]:.6e}", f"{err:.5f}",
                          f"{ratio:.5f}", (f"{meas:.5f}" if meas != "" else "")])
            flag = "" if err < 0.01 else "  <-- PARAM CHECK >1%"
            print(f"    {SIZE_LABEL[s]:3s} std={ff['standard']:.4e} sps={ff['sps']:.4e}"
                  f" | sps/std={ratio:.3f}"
                  + (f" meas={meas:.3f}" if meas != "" else "")
                  + f" | param_err={err*100:.3f}%{flag}")
    print(f"[save] {path}")


# ============================ main ================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(_REPO_ROOT / "figures" / "loss_vs_compute_wandb_cache.json"))
    ap.add_argument("--force-pull", action="store_true")
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"),
                    help="W&B entity; only needed for --force-pull (default: $WANDB_ENTITY)")
    ap.add_argument("--project", default=DEFAULT_PROJECT,
                    help=f"W&B project; only needed for --force-pull (default: {DEFAULT_PROJECT})")
    ap.add_argument("--outdir", default=str(_REPO_ROOT / "figures"))
    ap.add_argument("--crosscheck-csv", default=str(CSV_CROSSCHECK),
                    help="per-step history CSV to cross-check the final pre-decay NLLs against "
                         "(written by plot_validation_nll_history_wandb.py); skipped if absent")
    args = ap.parse_args()

    # Match the paper family's Times/usetex styling (small-caps + Times math via mathptmx) where a
    # latex binary exists, and fall back to plain Times + STIX math where it does not.
    enable_latex_if_available()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = pull_wandb(Path(args.cache), args.force_pull,
                         entity=args.entity, project=args.project)
    if not records:
        print("[fatal] no runs pulled")
        sys.exit(1)
    # restrict the whole deliverable to the in-scope archs (Standard vs SPS).
    records = [r for r in records if r["arch"] in ARCH_ORDER]

    # Published artifacts (manifest, points CSV) must carry the repo's clean run names, never
    # the pre-rename W&B spellings that the lookup fell back to -- those files ship in the
    # figure-data dataset. Normalising here rather than at pull time also cleans up a cache
    # that was written before this existed. The resolved W&B name is kept for provenance.
    for rec in records:
        rec.setdefault("wandb_name", rec.get("name"))
        rec["name"] = run_name(rec["arch"], rec["scale"], rec.get("budget", "20b"))

    curves, bands = build_curves(records)
    print(f"[build] {len(curves)} curves: "
          + ", ".join(f"{c.arch}/{c.scale}" for c in curves))

    csv_crosscheck(curves, Path(args.crosscheck_csv))
    write_flop_ratios(curves, records, outdir / "loss_vs_compute_flop_ratios.csv")
    write_points_csv(curves, outdir / "loss_vs_compute_points.csv")
    write_manifest(records, curves, outdir / "loss_vs_compute_manifest.txt")
    write_decay_gain(records, curves, outdir / "loss_vs_compute_decay_gain.csv")

    frontiers = {a: method_frontier(curves, a) for a in ARCH_ORDER}
    for a, fr in frontiers.items():
        if fr is None:
            continue
        if fr["kind"] == "powerlaw":
            print(f"[frontier:{a}] NLL = {fr['E']:.4f} + {fr['A']:.4f}*F^(-{fr['alpha']:.4f}) "
                  f"(power law; fit RMSE={fr['rmse']:.4f}, "
                  f"range {fr['lo']:.2e}..{fr['hi']:.2e} FLOPs)")
        else:
            print(f"[frontier:{a}] NLL = {fr['slope']:+.4f}*log10(FLOPs) + {fr['intercept']:.4f} "
                  f"(log-linear fallback; fit RMSE={fr['rmse']:.4f}, "
                  f"range {fr['lo']:.2e}..{fr['hi']:.2e} FLOPs)")
    # Crossover: no closed form for the power laws, so scan for the first sign change of
    # (standard - sps) over the union of both fit ranges plus the forward projection.
    fs, fp = frontiers.get("standard"), frontiers.get("sps")
    if fs is not None and fp is not None:
        lo = min(fs["lo"], fp["lo"])
        hi = max(fs["hi"], fp["hi"], FRONTIER_EXTRAP_MAX)
        gx = np.logspace(math.log10(lo), math.log10(hi), 4000)
        diff = np.asarray(fs["predict"](gx)) - np.asarray(fp["predict"](gx))  # + => SPS lower
        sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
        if len(sign_change):
            i = sign_change[0]
            x0l, x1l = math.log10(gx[i]), math.log10(gx[i + 1])
            xcl = x0l - diff[i] * (x1l - x0l) / (diff[i + 1] - diff[i])  # interp in log-F
            print(f"[frontier:crossover] SPS overtakes Standard at FLOPs=10^{xcl:.3f} "
                  f"= {10.0 ** xcl:.2e}")
        else:
            lead = "SPS" if diff[-1] > 0 else "Standard"
            print(f"[frontier:crossover] no crossover in range; {lead} leads throughout")

    # The shipped figure draws the empirical frontier: each architecture's min NLL over sizes
    # at every FLOP budget, plotted directly and stopping at the last measured point. The
    # fitted power law is still computed above -- the crossover diagnostic printed just above
    # depends on it -- but it is not drawn, so the figure asserts nothing past the data.
    make_figure(curves, bands, frontiers, "flops",
                outdir / "loss_vs_compute_flops.png",
                outdir / "loss_vs_compute_flops.pdf",
                frontier_mode="envelope")


if __name__ == "__main__":
    main()
