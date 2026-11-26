"""Pass family and run dataclasses to workers as ordinary command-line options."""

from dataclasses import fields
from importlib import import_module
import json
from pathlib import Path
import shlex
import sys
from typing import get_args, get_origin

PREFIX = "--cfg."


def _rows(module, table):
    records = getattr(module, table, ())
    if isinstance(records, dict):
        return records.items()
    return ((getattr(row, "name", "default"), row) for row in records)


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(type(value).__name__)


def worker_settings(families, selected=None):
    records = {}
    for family in families:
        module = import_module("experiments." + family)
        if hasattr(module, "family_config"):
            records[family] = module.family_config
        for table in ("experiment_configs", "job_configs"):
            rows = list(_rows(module, table))
            matches = [
                (name, row)
                for name, row in rows
                if selected is not None
                and (name == selected or getattr(row, "arm", None) == selected)
            ]
            for name, row in matches or rows:
                records[f"{family}.{table}.{name}"] = row
    return {
        group: {
            f.name.lower(): getattr(row, f.name)
            for f in fields(row)
            if f.init
            and f.name != "experiment_configs"
            and getattr(row, f.name) is not None
        }
        for group, row in records.items()
    }


def worker_command(command, settings):
    options = []
    for group, values in settings.items():
        for name, value in values.items():
            if isinstance(value, (dict, tuple, list, set, frozenset)):
                value = json.dumps(value, default=json_value, separators=(",", ":"))
            options.append(f"{PREFIX}{group}.{name}={value}")
    return command + (" " + shlex.join(options) if options else "")


def _collection(value, annotation):
    kind = get_origin(annotation) or annotation
    args = get_args(annotation)
    if kind is dict:
        if not args:
            return value
        key_type, value_type = args
        return {
            _collection(k, key_type): _collection(v, value_type)
            for k, v in value.items()
        }
    if kind in (tuple, list, frozenset, set):
        if not args:
            return kind(value)
        homogeneous = len(args) == 1 or args[-1] is Ellipsis
        return kind(
            _collection(v, args[0] if homogeneous else args[i])
            for i, v in enumerate(value)
        )
    return kind(value)


def parse_record(default, options):
    """Parse the declared dataclass, retaining this experiment's inherited defaults."""
    from transformers import HfArgumentParser

    class Parser(HfArgumentParser):
        @staticmethod
        def _parse_dataclass_field(parser, field):
            if field.name == "experiment_configs":
                return
            base = get_origin(field.type) or field.type
            if base in (dict, list, tuple, set, frozenset):
                parser.add_argument(
                    "--" + field.name,
                    dest=field.name,
                    type=lambda raw: _collection(json.loads(raw), field.type),
                )
            else:
                HfArgumentParser._parse_dataclass_field(parser, field)

    parser = Parser(type(default), allow_abbrev=False, add_help=False)
    by_name = {f.name.lower(): f.name for f in fields(default)}
    argv = []
    for name, value in options:
        if name not in by_name:
            raise ValueError(f"Unknown {type(default).__name__} field {name!r}")
        argv.append(f"--{by_name[name]}={value}")
    parser.set_defaults(
        **{f.name: getattr(default, f.name) for f in fields(default) if f.init}
    )
    for action in parser._actions:
        if action.dest in vars(default):
            action.required = False
    (record,) = parser.parse_args_into_dataclasses(argv, look_for_args_file=False)
    return record


def read_worker_arguments(argv):
    """Install the requested dataclasses before importing the native worker."""
    groups = {}
    remaining = []
    for arg in argv:
        if not arg.startswith(PREFIX):
            remaining.append(arg)
            continue
        key, sep, value = arg[len(PREFIX) :].partition("=")
        if not sep:
            raise ValueError(f"Expected {PREFIX}group.field=value")
        group, name = key.rsplit(".", 1)
        groups.setdefault(group, []).append((name, value))
    for group, options in groups.items():
        family, *rest = group.split(".", 2)
        module = import_module("experiments." + family)
        if not rest:
            module.family_config = parse_record(module.family_config, options)
            continue
        table, name = rest
        if table not in ("experiment_configs", "job_configs"):
            raise ValueError(f"Unknown experiment table {table}")
        rows = getattr(module, table)
        for key, row in rows.items() if isinstance(rows, dict) else enumerate(rows):
            if (
                str(key) if isinstance(rows, dict) else getattr(row, "name", "default")
            ) == name:
                rows[key] = parse_record(row, options)
                break
        else:
            raise ValueError(f"Unknown experiment {group}")
    argv[:] = remaining


def worker_child(command):
    """Pass the parent's CLI settings through local and torchrun child processes."""
    if not any(
        str(word).startswith("scripts/") and str(word).endswith(".py")
        for word in command
    ):
        return command
    return [*command, *(arg for arg in sys.orig_argv if arg.startswith(PREFIX))]
