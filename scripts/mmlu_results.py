"""Durable sample stream recovery (stdlib-only, usable in CPU CI)."""
import json


def recover_samples(path):
    saved = []
    if not path.exists():
        return saved
    with path.open('rb+') as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                # Treat a missing newline as an uncommitted row, even if JSON is valid.
                stream.truncate(start)
                break
            try:
                saved.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError('Corrupt committed sample row') from error
    return saved
