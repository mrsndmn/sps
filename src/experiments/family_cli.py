"""Pass the family dataclasses to a native worker as explicit Hugging Face CLI flags."""

import json
import os
import shlex
import sys
from dataclasses import fields, make_dataclass
from importlib import import_module
from pathlib import Path

from experiments.environment import PYTHON
from experiments.family_arguments import definition_rows


def wire_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (frozenset, set)):
        return sorted(value)
    raise TypeError(f"Unsupported experiment argument {type(value).__name__}")


def arguments(names, selected=None):
    """Use qualified flags so that independent workloads can both declare a seed."""
    groups = {}
    for name in names:
        module = import_module("experiments." + name)
        config = getattr(module, "family_config", None)
        values = {}
        if config is not None:
            values.update(
                (f.name, getattr(config, f.name))
                for f in fields(config)
                if f.init and f.name.isupper()
            )
        for table_name in ("experiment_configs", "job_configs"):
            entries = list(enumerate(definition_rows(module, table_name)))
            matches = [
                (i, row) for i, row in entries if getattr(row, "name", None) == selected
            ]
            for index, row in matches or entries:
                for field in fields(row):
                    if field.init:
                        values[f"{table_name}__{index}__{field.name}"] = getattr(
                            row, field.name
                        )
        groups[name] = values
    return groups


def flag(group, name):
    return f"{group}__{name.lower()}"


def encode(groups):
    argv = []
    for group, values in groups.items():
        for name, value in values.items():
            raw = (
                str(value)
                if isinstance(value, (str, float, int, Path))
                else json.dumps(value, default=wire_value, separators=(",", ":"))
            )
            argv.extend(("--" + flag(group, name), raw))
    return argv


def parse(groups, argv):
    from transformers import HfArgumentParser

    schema = []
    for group, values in groups.items():
        for name, value in values.items():
            kind = type(value) if type(value) in (float, int, bool, str) else str
            schema.append((flag(group, name), kind))
    parser = HfArgumentParser(
        make_dataclass("FamilyArguments", schema), allow_abbrev=False
    )
    (parsed,) = parser.parse_args_into_dataclasses(args=argv, look_for_args_file=False)
    overrides = {}
    for group, values in groups.items():
        data = {}
        for name, default in values.items():
            value = getattr(parsed, flag(group, name))
            decoded = (
                value
                if isinstance(default, (str, float, int, Path))
                else json.loads(value)
            )
            if "__" in name:
                table, index, key = name.split("__", 2)
                data.setdefault(table, {}).setdefault(index, {})[key] = decoded
            else:
                data[name] = decoded
        overrides["experiments." + group] = data
    return overrides


def worker_command(command, names, selected=None):
    if not names:
        return command
    groups = arguments(names, selected)
    return shlex.join(
        [
            str(PYTHON),
            "-m",
            "experiments.family_cli",
            ",".join(groups),
            selected or "-",
            *encode(groups),
            "--",
            "/bin/bash",
            "-c",
            command,
        ]
    )


def main():
    separator = sys.argv.index("--")
    names = sys.argv[1].split(",")
    values = parse(arguments(names, sys.argv[2]), sys.argv[3:separator])
    os.environ["SPS_FAMILY_ARGUMENTS"] = json.dumps(values, separators=(",", ":"))
    command = sys.argv[separator + 1 :]
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
