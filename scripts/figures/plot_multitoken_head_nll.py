#!/usr/bin/env python3
"""Render the multi-token-ahead linear-probe NLL figure: per look-ahead offset
``k`` (x) vs. probe validation NLL (y), one column per scale, a single row.

Each released checkpoint is probed with frozen backbone + per-offset LM heads
(see ``scripts/analysis/multitoken_head_nll.py``). Per scale:

* ``full-attention`` (Standard): its single ``main`` stream (solid).
* ``sps-w64``: ``prediction`` (solid) and ``state`` (dashed) streams.
* ``delayed-state-w64``: ``prediction`` (solid) and ``state`` (dashed) streams.

**Color = model family** (``_method_color``: Standard black, SPS green, Delayed
State orange); **linestyle = stream** (solid = prediction/readout/``main``,
dashed = state/input).

Each panel plots $\\Delta\\ell$ = (stream NLL - Standard's readout NLL at the same ``k``),
so Standard is the flat reference at 0 and *negative = predicts that k-ahead token better
than Standard* (the persistent-window delta-NLL sign convention).

**The y-axis is symlog** (``linthresh=0.1``) because the two streams differ by ~65x in
effect size and cannot share a linear axis: prediction-stream deltas span -0.067..+0.004
nats (bootstrap CI half-width ~0.007-0.010, so the effect is 4-8x its CI at the larger
scales), while state-stream deltas span +0.11..+4.57. The linear zone below the break
resolves the prediction curves; the log zone above it carries the state curves without
flattening them. A dotted rule marks the linear/log break so the scale change is explicit.

Plotting deltas rather than absolutes is what makes one row work at all: on an absolute
axis the three readout curves coincide to within ~0.05 nats -- sub-pixel on a 3-8 nat
axis -- so Standard's black line simply hid the SPS and Delayed State readouts.

NB: for ``delayed_state`` the retained state lives in the odd (prediction) slot, so its
dashed "state" curve is the *windowed input* slot, not a carried-state stream -- which is
why the dashed legend entry reads "Input Stream" (the even slot) rather than naming a
carried state: the same dashed line means a structurally different thing for each family.

Reads per-scale ``<root>/<scale>/summary.json`` (written by
``run_multitoken_head_nll.sh merge``); falls back to the individual
``<root>/<scale>/<variant>.json`` files if no summary is present. Run it as::

    uv run python scripts/figures/plot_multitoken_head_nll.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# figure_common selects the Agg backend + serif rcParams (and pins MPLCONFIGDIR off
# NFS) before matplotlib.pyplot is imported below, and provides the shared styling
# constants + plot helpers.
from plotting.figure_common import (
    AXIS_LABEL_FONT_SIZE,
    LINEWIDTH,
    PANEL_GRID_WSPACE,
    PANEL_TITLE_FONT_SIZE,
    PLOT_DPI,
    _method_color,
    _method_label,
    _paired_image_paths,
    _plot_line,
    _read_json,
)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator
import numpy as np

from plotting.style import (
    LEGEND_TEXT_COLOR,
    _apply_axes_background,
    bf,
    enable_latex_if_available,
    sc,
)


DEFAULT_SCALES = ("xs", "s", "m", "l", "xl")
DEFAULT_ROOT = Path("outputs/multitoken_head_nll")
DEFAULT_OUTPUT_PATH = Path("figures/multitoken_head_nll.png")
FIXED_FIGURE_WIDTH = 17.0

# Variant -> (family/kind for coloring, sliding window for the method lookup). The
# family/window pair must resolve through figure_common._method_color's fallback
# table (full_attention/None, sps/64, delayed_state/64).
VARIANT_FAMILY_WINDOW: dict[str, tuple[str, int | None]] = {
    "full-attention": ("full_attention", None),
    "sps-w64": ("sps", 64),
    "delayed-state-w64": ("delayed_state", 64),
}
# Draw order (Standard first so its black curve sits under the colored families).
VARIANT_ORDER = ("full-attention", "sps-w64", "delayed-state-w64")

# Stream -> linestyle. Solid = the natively-read prediction/readout slot (and the
# standard model's single stream); dashed = the state/input slot.
STREAM_LINESTYLE = {"main": "-", "prediction": "-", "state": (0, (6, 4))}
STREAM_ZORDER = {"main": 6, "prediction": 6, "state": 5}
CI_ALPHA = 0.16
LEGEND_FONT_SIZE = 19

# Single-row layout: one panel per scale, absolute probe NLL (NLL vs k) for every stream.
#
# The label/legend stack below the panels is sized in points (fonts are absolute), so the
# figure-fraction constants below are the two-row values rescaled to preserve the SAME
# absolute spacing in inches: the old layout reserved 0.270 * 9.8 = 2.65in below the panels
# and gave each row ~3.3in, so one row of panels + the same 2.65in stack is ~6.2in tall.
FIGURE_HEIGHT_1ROW = 6.2
PANEL_TOP = 0.945
# Bottom of the panel row (figure coords, identical for 1 or 5 columns). Below it, well
# separated so they never collide: x tick labels, then the x-axis label, then the two legends.
PANEL_BOTTOM = 0.427
# Vertical stack below the panels, top->bottom: x-tick labels, x-axis label, family legend,
# stream legend. The x-axis label ("Tokens ahead k") is 26pt bold, so it needs a
# wide gap above the family legend or the "k" descender collides with the legend row; it sits
# just under the ticks. The legends are pushed toward the bottom edge
# (bbox_inches="tight" absorbs any overhang below the stream legend).
X_AXIS_LABEL_Y = 0.316
FAMILY_LEGEND_ANCHOR = (0.5, 0.169)
STREAM_LEGEND_ANCHOR = (0.5, 0.082)
# Y-axis label x-position (axes fraction). Pushed left of the signed two-decimal delta
# ticks; bbox_inches="tight" absorbs the extra left extent at save.
Y_LABEL_COORD_X = -0.30

# --- symlog delta axis ----------------------------------------------------------------
# Below +/-LINTHRESH the axis is linear, above it log. 0.1 sits just above the largest
# prediction-stream |delta| (0.067) and just below the smallest state-stream delta (0.11),
# so each stream lands wholly in the zone that suits it.
SYMLOG_LINTHRESH = 0.1
# Decades of visual height given to each half of the linear zone. At 1.2 the linear band
# (+/-0.1) takes ~58% of the axis and the 0.1->5 log span ~42%, so the prediction curves
# get the majority of the height without squashing the state curves into the top edge.
SYMLOG_LINSCALE = 1.2
# Explicit ticks: symlog's auto-locator emits only 0 and powers of ten, which leaves the
# linear zone (where the whole prediction story lives) unlabelled.
DELTA_YTICKS = (-0.06, -0.03, 0.0, 0.03, 0.06, 0.1, 0.3, 1.0, 3.0)
# Headroom multiplier on the extreme deltas so curves/CI bands never touch the spines.
DELTA_PAD = 1.25


def _load_scale_models(root: Path, scale: str) -> list[dict[str, Any]]:
    """Return the per-variant payloads for a scale.

    Prefer the merged ``<scale>/summary.json`` (``{"scale", "models": [...]}``); fall
    back to the individual ``<scale>/<variant>.json`` compute outputs so the plot
    works before a merge has run. Returns ``[]`` when a scale has no data yet."""
    summary = root / scale / "summary.json"
    if summary.exists():
        return list(_read_json(summary).get("models", []))
    models: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        path = root / scale / f"{variant}.json"
        if path.exists():
            models.append(_read_json(path))
    return models


def _ci_halfwidth(row: dict[str, Any]) -> float:
    return 0.5 * (float(row.get("ci_hi", row["nll"])) - float(row.get("ci_lo", row["nll"])))


def _standard_by_k(models: dict[str, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Standard's ``main``-stream rows keyed by k -- the per-k reference every delta uses."""
    std = models.get("full-attention")
    if std is None:
        return {}
    return {int(r["k"]): r for r in std.get("results", []) if r["stream"] == "main"}


def _deltas_for(
    model: dict[str, Any], stream: str, std_by_k: dict[int, dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ks, delta, ci_halfwidth)`` for one stream vs Standard's readout at the same k.

    The CI of a difference of two independent estimates adds half-widths in quadrature."""
    rows = sorted(
        (r for r in model.get("results", []) if r["stream"] == stream and int(r["k"]) in std_by_k),
        key=lambda r: int(r["k"]),
    )
    if not rows:
        return np.empty(0), np.empty(0), np.empty(0)
    ks = np.asarray([int(r["k"]) for r in rows], dtype=float)
    delta = np.asarray([float(r["nll"]) - float(std_by_k[int(r["k"])]["nll"]) for r in rows])
    hw = np.asarray([
        (_ci_halfwidth(r) ** 2 + _ci_halfwidth(std_by_k[int(r["k"])]) ** 2) ** 0.5 for r in rows
    ])
    return ks, delta, hw


def _delta_bounds(
    models_by_scale: dict[str, list[dict[str, Any]]], scales: tuple[str, ...]
) -> tuple[float, float]:
    """Y-range over every stream's delta (plus its CI band), forced to include 0 and the
    linear/log break so the symlog break line is always on-axis."""
    lo_vals: list[float] = [-SYMLOG_LINTHRESH]
    hi_vals: list[float] = [SYMLOG_LINTHRESH]
    for scale in scales:
        models = {m.get("variant"): m for m in models_by_scale.get(scale, [])}
        std_by_k = _standard_by_k(models)
        if not std_by_k:
            continue
        for variant in ("sps-w64", "delayed-state-w64"):
            model = models.get(variant)
            if model is None:
                continue
            for stream in ("prediction", "state"):
                _, delta, hw = _deltas_for(model, stream, std_by_k)
                if delta.size:
                    lo_vals.append(float((delta - hw).min()))
                    hi_vals.append(float((delta + hw).max()))
    return (min(lo_vals) * DELTA_PAD, max(hi_vals) * DELTA_PAD)


def _format_delta_tick(value: float, _pos: int | None = None) -> str:
    """Two decimals inside the linear zone (deltas there are ~0.01-0.07), one decimal in
    the log zone (0.3/1.0/3.0), and a bare ``0`` for the Standard reference line."""
    if abs(value) < 1e-12:
        return "0"
    return f"{value:.2f}" if abs(value) < 0.15 else f"{value:.1f}"


def render(
    *,
    models_by_scale: dict[str, list[dict[str, Any]]],
    scales: tuple[str, ...],
    k_max: int,
    output_path: Path,
) -> Path:
    from plotting.figure_common import _style_ticks

    y_lo, y_hi = _delta_bounds(models_by_scale, scales)
    x_min, x_max = 0.6, k_max + 0.4
    std_color = _method_color(*VARIANT_FAMILY_WINDOW["full-attention"])

    # sharey ties every panel to one delta axis so the columns are directly comparable.
    fig, axes = plt.subplots(
        1, len(scales), figsize=(FIXED_FIGURE_WIDTH, FIGURE_HEIGHT_1ROW),
        sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes).reshape(len(scales))

    for col, scale in enumerate(scales):
        ax_t = axes[col]
        _apply_axes_background(ax_t, x_min=x_min, x_max=x_max, y_min=y_lo, y_max=y_hi)
        # symlog must be set before the limits so the locator sees the final scale.
        ax_t.set_yscale("symlog", linthresh=SYMLOG_LINTHRESH, linscale=SYMLOG_LINSCALE)
        ax_t.set_ylim(y_lo, y_hi)

        models = {m.get("variant"): m for m in models_by_scale.get(scale, [])}
        std_by_k = _standard_by_k(models)

        # --- the linear/log break, drawn under everything -------------------------
        ax_t.axhline(
            SYMLOG_LINTHRESH, color=LEGEND_TEXT_COLOR, linewidth=1.1, alpha=0.45,
            linestyle=(0, (2, 3)), zorder=1.5,
        )

        # --- Standard: the flat 0 reference (its own delta) -----------------------
        if std_by_k:
            ks0 = np.asarray(sorted(std_by_k), dtype=float)
            ax_t.plot(ks0, np.zeros_like(ks0), color=std_color, linewidth=LINEWIDTH * 0.6,
                      alpha=0.9, solid_capstyle="round", zorder=2)

        # --- both streams of each predict-token family, as deltas vs Standard -----
        for variant in ("sps-w64", "delayed-state-w64"):
            model = models.get(variant)
            if model is None or not std_by_k:
                continue
            color = _method_color(*VARIANT_FAMILY_WINDOW[variant])
            for stream in ("prediction", "state"):
                ks, delta, hw = _deltas_for(model, stream, std_by_k)
                if not delta.size:
                    continue
                z = STREAM_ZORDER.get(stream, 5)
                ax_t.fill_between(ks, delta - hw, delta + hw, color=color, alpha=CI_ALPHA,
                                  linewidth=0, zorder=z - 2)
                _plot_line(ax_t, ks, delta, color=color,
                           linestyle=STREAM_LINESTYLE.get(stream, "-"), zorder=z)

        ax_t.set_title(
            bf(scale.upper()),
            fontsize=PANEL_TITLE_FONT_SIZE,
            fontweight="bold",
            color=LEGEND_TEXT_COLOR,
            pad=9,
        )
        ax_t.xaxis.set_major_locator(MultipleLocator(1))
        ax_t.set_xlim(x_min, x_max)
        # Explicit delta ticks (symlog's locator would label only 0 and powers of ten,
        # leaving the linear zone -- where the prediction story lives -- unlabelled).
        ticks = [t for t in DELTA_YTICKS if y_lo <= t <= y_hi]
        ax_t.yaxis.set_major_locator(FixedLocator(ticks))
        ax_t.yaxis.set_major_formatter(FuncFormatter(_format_delta_tick))
        _style_ticks(ax_t)
        ax_t.minorticks_off()

    axes[0].set_ylabel(
        _bold_math(r"$\Delta \ell$"),
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="bold",
        color=LEGEND_TEXT_COLOR,
    )
    axes[0].yaxis.set_label_coords(Y_LABEL_COORD_X, 0.5)
    fig.text(
        0.5,
        X_AXIS_LABEL_Y,
        _bold_math("Tokens ahead $k$"),
        ha="center",
        va="center",
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="bold",
        color=LEGEND_TEXT_COLOR,
    )

    # Legend 1: family (color). Legend 2: stream (linestyle).
    family_handles = [
        Line2D([0], [0], color=_method_color(fam, win), linewidth=LINEWIDTH, label=sc(_method_label(fam, win)))
        for variant in VARIANT_ORDER
        for (fam, win) in (VARIANT_FAMILY_WINDOW[variant],)
    ]
    stream_handles = [
        Line2D([0], [0], color=LEGEND_TEXT_COLOR, linewidth=LINEWIDTH, linestyle="-",
               label=sc("Prediction Stream")),
        Line2D([0], [0], color=LEGEND_TEXT_COLOR, linewidth=LINEWIDTH, linestyle=(0, (6, 4)),
               label=sc("Input Stream")),
    ]
    family_legend = fig.legend(
        handles=family_handles, loc="lower center", ncol=len(family_handles), frameon=False,
        bbox_to_anchor=FAMILY_LEGEND_ANCHOR, fontsize=LEGEND_FONT_SIZE,
        handlelength=2.35, columnspacing=1.4, handletextpad=0.5,
    )
    for text in family_legend.get_texts():
        text.set_color(LEGEND_TEXT_COLOR)
    fig.add_artist(family_legend)
    stream_legend = fig.legend(
        handles=stream_handles, loc="lower center", ncol=len(stream_handles), frameon=False,
        bbox_to_anchor=STREAM_LEGEND_ANCHOR, fontsize=LEGEND_FONT_SIZE,
        handlelength=2.9, columnspacing=1.4, handletextpad=0.5,
    )
    for text in stream_legend.get_texts():
        text.set_color(LEGEND_TEXT_COLOR)

    fig.subplots_adjust(
        left=0.10, right=0.985, bottom=PANEL_BOTTOM, top=PANEL_TOP,
        wspace=PANEL_GRID_WSPACE,
    )
    png_path, pdf_path = _paired_image_paths(output_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path


def _bold_math(label: str) -> str:
    r"""Return ``label`` with its math spans bolded so they match the bold text labels.

    The panel titles/legend render bold (plain text + ``fontweight="bold"``), but the
    axis labels are *math* (``$\ell$``, ``$\Delta\ell$``, the ``$k$`` in "Tokens ahead
    $k$"). Under usetex ``bf()`` wraps in ``\textbf`` and mathptmx carries that weight
    into the math -- so we keep ``bf()`` there and the paper output stays byte-identical
    to the persistent-window sibling. Without LaTeX, neither
    ``bf()`` nor ``fontweight`` can bold math, so ``$\ell$``/``$\Delta\ell$`` came out thin
    and weak next to the heavy text -- the visible mismatch. matplotlib's own mathtext *does*
    honor ``\boldsymbol{...}`` (no LaTeX needed), so wrap each ``$...$`` span in it to give the
    math the same weight as the surrounding labels."""
    if plt.rcParams.get("text.usetex", False):
        return bf(label)
    return re.sub(r"\$(.+?)\$", lambda m: rf"$\boldsymbol{{{m.group(1)}}}$", label)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="dir containing <scale>/{summary,variant}.json")
    parser.add_argument("--scales", nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    # ``sc()``/``bf()`` no-op when usetex is off, so the no-LaTeX fallback just loses the
    # ``\textsc`` small-caps and renders with the serif rcParams ``figure_common`` sets on import.
    enable_latex_if_available()
    models_by_scale = {scale: _load_scale_models(args.root, scale) for scale in args.scales}
    present = tuple(scale for scale in args.scales if models_by_scale.get(scale))
    if not present:
        raise SystemExit(f"No data found under {args.root} for scales {args.scales}")
    if len(present) < len(args.scales):
        missing = [s for s in args.scales if s not in present]
        print(f"[plot] no data for scales {missing}; plotting {list(present)}")
    png = render(
        models_by_scale=models_by_scale,
        scales=present,
        k_max=int(args.k_max),
        output_path=args.output,
    )
    for path in _paired_image_paths(png):
        print(path)


if __name__ == "__main__":
    main()
