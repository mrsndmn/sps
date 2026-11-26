"""Shared location for analysis products, benchmark results and table exports."""
import os
from pathlib import Path

OUTPUT_ROOT = Path(os.environ.get(
    "SPS_OUTPUT_ROOT", "/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/sps-artifacts/outputs"
)).expanduser().resolve()


if __name__ == "__main__":
    print(OUTPUT_ROOT)
