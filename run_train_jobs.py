"""Submit data/probes or the paired four-GPU training jobs; --dry never submits."""
from scripts.mlspace_launch import main

if __name__ == '__main__':
    raise SystemExit(main())
