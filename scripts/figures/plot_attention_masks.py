#!/usr/bin/env python3
"""Attention-mask schematic for Standard, SPS, Delayed State, Reverse SPS, and 2x Memory.

Each panel shows which keys (columns) a query (rows) may attend, on the interleaved
length-2T sequence of the predict-token families. Two facts the panels are built to make
visible, because they are what distinguishes the variants:

* which parity carries the persistent keys, and which is held only in the sliding window;
* which slot the next-token loss is read from.

Delayed State and Reverse SPS share an identical mask and differ *only* in the readout slot,
so the mask alone does not separate them.

    uv run python scripts/figures/plot_attention_masks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plotting.mpl_cache  # noqa: F401

import argparse

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif",
                   "STIXGeneral", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
})
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

from plotting.style import (
    LEGEND_TEXT_COLOR, PLOT_SPINE_COLOR, bf, enable_latex_if_available, sc,
)

T = 4          # real tokens drawn
W = 1          # sliding window, in token units
BLOCKED   = "#f2ece1"   # not attended
PERSIST   = "#2f6f9f"   # persistent key (never evicted)
WINDOWED  = "#e08a3c"   # ephemeral key, kept only within the window
READOUT   = "#1a1a1a"   # marker colour for the slot the loss is read from
GRIDLINE  = "#c9bfae"
# Blank strip left of the matrix holding the readout markers, as a FRACTION of the
# sequence length. A fixed width would eat a larger share of the short Standard panel
# than of the doubled ones, shrinking its matrix; scaling it keeps every panel's data
# aspect at 1.15, so all five matrices render at identical size in equal axes boxes.
GUTTER_FRAC = 0.15

# variant -> (persistent parity, readout parity, window, doubled?)
#   parity 0 = input slot (even), 1 = <predict> slot (odd)
VARIANTS = [
    ("Standard",      None, None, None, False, (r"state $x$", r"predict $x$")),
    ("SPS",           0,    1,    W,    True,   (r"state $x$", r"predict $\rho$")),
    ("Delayed State", 1,    1,    W,    True,   (r"state $\rho$", r"predict $\rho$")),
    ("Reverse SPS",   1,    0,    W,    True,   (r"state $\rho$", r"predict $x$")),
    ("2x Memory",     0,    1,    T,    True,   (r"state $x,\rho$", r"predict $\rho$")),
]


def mask_for(persist_parity, window, doubled):
    """Return (n, n) int array: 0 blocked, 1 persistent key, 2 windowed key."""
    if not doubled:
        n = T
        m = np.zeros((n, n), dtype=int)
        for q in range(n):
            for k in range(q + 1):
                m[q, k] = 1
        return m
    n = 2 * T
    m = np.zeros((n, n), dtype=int)
    for q in range(n):
        for k in range(q + 1):                      # causal on the doubled sequence
            if k % 2 == persist_parity:
                m[q, k] = 1                          # persistent: always visible
            elif (q // 2) - (k // 2) <= window:
                m[q, k] = 2                          # ephemeral: only inside the window
    if window >= T:      # window spans the context, so no key is ever evicted
        m[m == 2] = 1
    return m


def render(output_path: Path) -> Path:
    enable_latex_if_available()
    # Two rows of three. Five variants leave the sixth cell free, which the legend takes,
    # and the wider panels are what make this readable at appendix size.
    fig, axgrid = plt.subplots(2, 3, figsize=(14.0, 9.4))
    axes = axgrid.ravel()

    for ax, (name, pp, rp, win, doubled, subtitle) in zip(axes, VARIANTS):
        m = mask_for(pp, win, doubled)
        n = m.shape[0]
        rgb = np.empty(m.shape + (3,))
        for val, colour in ((0, BLOCKED), (1, PERSIST), (2, WINDOWED)):
            rgb[m == val] = matplotlib.colors.to_rgb(colour)
        ax.imshow(rgb, interpolation="nearest", origin="upper")

        for e in range(n + 1):                       # cell grid, matrix only
            ax.plot([-0.5, n - 0.5], [e - 0.5, e - 0.5], color=GRIDLINE, lw=0.6, zorder=3)
            ax.plot([e - 0.5, e - 0.5], [-0.5, n - 0.5], color=GRIDLINE, lw=0.6, zorder=3)

        labels = ([f"$x_{{{i+1}}}$" for i in range(T)] if not doubled else
                  [f"$x_{{{i//2+1}}}$" if i % 2 == 0 else f"$\\rho_{{{i//2+1}}}$"
                   for i in range(n)])
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=17)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=17)
        ax.tick_params(length=0)

        readout = range(n) if rp is None else [i for i in range(n) if i % 2 == rp]
        for r in readout:                            # mark rows that emit the loss
            ax.plot(-0.5 - GUTTER_FRAC * n / 2, r, marker=">", color=READOUT,
                    markersize=9, clip_on=False, linestyle="none")
        gutter = GUTTER_FRAC * n
        ax.set_xlim(-0.5 - gutter, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
        ax.set_anchor("N")
        for sp in ax.spines.values():                # box the matrix, not the gutter
            sp.set_visible(False)
        ax.add_patch(Rectangle((-0.5, -0.5), n, n, fill=False,
                               edgecolor=PLOT_SPINE_COLOR, lw=1.2, zorder=5))
        cx = (0.5 + GUTTER_FRAC) / (1.0 + GUTTER_FRAC)   # centre on the matrix, not the axes
        ax.set_title(bf(sc(name)), fontsize=22, color=LEGEND_TEXT_COLOR, pad=46, x=cx)
        state_txt, pred_txt = subtitle
        ax.text(cx, 1.042, f"{state_txt} $\\cdot$ {pred_txt}", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=17, color=LEGEND_TEXT_COLOR)

    # the free sixth cell carries the legend; axis names are left to the caption
    legend_ax = axes[len(VARIANTS)]
    legend_ax.axis("off")
    handles = [
        Patch(facecolor=PERSIST,  label=sc("Persistent key")),
        Patch(facecolor=WINDOWED, label=sc("Windowed key")),
        Patch(facecolor=BLOCKED,  label=sc("Not attended")),
        plt.Line2D([0], [0], marker=">", color=READOUT, lw=0,
                   markersize=12, label=sc("Loss read here")),
    ]
    leg = legend_ax.legend(handles=handles, loc="center", frameon=False, fontsize=19,
                           handlelength=1.8, labelspacing=1.5, borderpad=1.0)
    for t in leg.get_texts():
        t.set_color(LEGEND_TEXT_COLOR)

    fig.subplots_adjust(left=0.045, right=0.985, top=0.92, bottom=0.02,
                        wspace=0.30, hspace=0.46)
    png, pdf = output_path.with_suffix(".png"), output_path.with_suffix(".pdf")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[save] {png}\n[save] {pdf}")
    return png


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("figures/attention_masks.png"))
    render(ap.parse_args().output)
