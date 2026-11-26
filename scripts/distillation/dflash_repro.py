"""Run dflash repro."""

from pathlib import Path
import sys


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from experiments.family_cli import read_worker_arguments

    read_worker_arguments(sys.argv)
    from workloads.distillation.dflash_repro import main

    main()
