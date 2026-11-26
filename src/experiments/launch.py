"""A public run name and the native launcher command for one phase."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Launch:
    name: str
    phase: str
    script: str
    args: tuple[str, ...] = ()
    options: tuple[str, ...] = ()
    required: tuple[str, ...] = ()


def filtered(rows, phase, script, *args, options=(), required=()):
    for row in rows:
        yield Launch(
            row.name, phase, script, (*args, "--filter", row.name), options, required
        )


def run_filter(parser, rows):
    by_name = {row.name: row for row in rows}
    parser.add_argument(
        "--filter",
        choices=by_name,
        required=len(by_name) != 1,
        default=next(iter(by_name), None),
    )
    return by_name


def matching_runs(rows, pattern):
    """Prefer a complete run name, while retaining native batch substring filters."""
    rows = list(rows)
    if pattern is None:
        return rows
    exact = [row for row in rows if row.name == pattern]
    return exact or [row for row in rows if pattern in row.name]
