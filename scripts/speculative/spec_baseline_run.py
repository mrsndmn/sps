"""Run spec baseline run."""

from pathlib import Path
import sys


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from workloads.speculative.spec_baseline_run import main

    main()
