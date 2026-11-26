"""Run multitoken head nll io."""

from pathlib import Path
import sys


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from analysis.multitoken_head_nll_io import main

    main()
