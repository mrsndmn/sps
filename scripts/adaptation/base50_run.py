"""Run base50 run."""

from pathlib import Path
import sys


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from workloads.adaptation.base50_run import main

    main()
