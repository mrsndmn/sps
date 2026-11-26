"""Submit data/probes or the paired four-GPU training jobs; --dry never submits."""
from experiment_defs.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
