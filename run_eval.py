"""Submit checkpoint-scoped evaluation jobs for finished pilot models."""
from scripts.mlspace_launch import main

if __name__ == '__main__':
    raise SystemExit(main('eval'))
