"""Run persistent window nll analysis."""

from pathlib import Path
import sys


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from analysis.persistent_window_nll_analysis import main

    main()
