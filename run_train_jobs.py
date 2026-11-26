"""Submit data/probes or the paired four-GPU training jobs; --dry never submits."""
from scripts.named.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
