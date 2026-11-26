"""Run vllm sps run."""

from pathlib import Path
import sys


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from workloads.serving.vllm_sps_run import main

    main()
