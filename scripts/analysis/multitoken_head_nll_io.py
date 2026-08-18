#!/usr/bin/env python3
"""Merge helper for the multi-token-ahead linear-probe NLL analysis.

Each compute task writes one ``<root>/<scale>/<variant>.json`` payload
(``multitoken_head_nll.py``). This folds a scale's per-variant payloads into
``<root>/<scale>/summary.json`` -- ``{"scale": ..., "models": [payload, ...]}`` -- which
``scripts/figures/plot_multitoken_head_nll.py`` reads, and prints a per-scale eyeball
table: readout-stream NLL at ``k=1`` (comparable to the model's own tied head and to the
release ``final_val_loss``) and at ``k=k_max``, plus the input-stream ``k=1``.

The plot falls back to the individual ``<variant>.json`` files when no summary is
present, so merging is a convenience (one file per scale, stable field order), not a
requirement.

Trivial CPU work -- a handful of small JSON files. Run it after the compute array
finishes::

    uv run python -m scripts.analysis.multitoken_head_nll_io --root outputs/multitoken_head_nll
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_SCALES = ("xs", "s", "m", "l", "xl")
DEFAULT_VARIANTS = ("full-attention", "sps-w64", "delayed-state-w64")


def readout_stream(kind: str) -> str:
    """The stream a model natively takes logits from: ``main`` for the standard model,
    the ``<predict>``-token slot for the predict-token families."""
    return "main" if kind == "full_attention" else "prediction"


def load_variant_payloads(scale_dir: Path, variants: tuple[str, ...]) -> list[dict[str, Any]]:
    """Per-variant payloads present for one scale, in ``variants`` order."""
    payloads = []
    for variant in variants:
        path = scale_dir / f"{variant}.json"
        if path.exists():
            payloads.append(json.loads(path.read_text()))
    return payloads


def merge_scale(root: Path, scale: str, variants: tuple[str, ...]) -> Path | None:
    """Write ``<root>/<scale>/summary.json`` for one scale; None if it has no payloads."""
    scale_dir = root / scale
    models = load_variant_payloads(scale_dir, variants)
    if not models:
        return None
    summary_path = scale_dir / "summary.json"
    summary_path.write_text(json.dumps({"scale": scale, "models": models}, indent=2))
    return summary_path


def summary_line(model: dict[str, Any]) -> str:
    """One eyeball line per model: readout k=1/k_max, input-stream k=1, and the two
    anchors the k=1 readout should sit next to (the native tied head, and the release
    ``final_val_loss``)."""
    nll_by_stream_k = {(row["stream"], row["k"]): row["nll"] for row in model.get("results", [])}
    readout = readout_stream(model["kind"])
    k_max = model.get("k_max", 8)
    first = nll_by_stream_k.get((readout, 1))
    last = nll_by_stream_k.get((readout, k_max))
    state = nll_by_stream_k.get(("state", 1))
    parts = [f"    {model['variant']:<20}", f"readout k1={first:.3f}" if first is not None else "readout k1=NA"]
    if last is not None:
        parts.append(f"k{k_max}={last:.3f}")
    if state is not None:
        parts.append(f"| input k1={state:.3f}")
    parts.append(
        f"| native_k1={model.get('native_lmhead_k1_nll')} "
        f"final_val={model.get('release_final_val_loss')}"
    )
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("outputs/multitoken_head_nll"))
    parser.add_argument("--scales", nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    args = parser.parse_args()

    variants = tuple(args.variants)
    for scale in args.scales:
        summary_path = merge_scale(args.root, scale, variants)
        if summary_path is None:
            print(f"[merge] {scale}: no variant payloads, skipping")
            continue
        models = json.loads(summary_path.read_text())["models"]
        print(f"[merge] {scale}: {len(models)} model(s) -> {summary_path}")
        for model in models:
            print(summary_line(model))


if __name__ == "__main__":
    main()
