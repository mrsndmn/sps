"""Pass the family dataclasses to a native worker as explicit Hugging Face CLI flags."""

from dataclasses import fields, make_dataclass
from importlib import import_module
import json
import os
from pathlib import Path
import shlex
import sys

from experiments.environment import PYTHON


def wire_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (frozenset, set)):
        return sorted(value)
    raise TypeError(f"Unsupported experiment argument {type(value).__name__}")


def arguments(names):
    """Use qualified flags so that independent workloads can both declare a seed."""
    groups = {}
    for name in names:
        config = import_module("experiments." + name).family_config
        groups[name] = {
            f.name: getattr(config, f.name)
            for f in fields(config)
            if f.init and f.name.isupper()
        }
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
            data[name] = (
                value
                if isinstance(default, (str, float, int, Path))
                else json.loads(value)
            )
        overrides["experiments." + group] = data
    return overrides


def worker_command(command, names):
    if not names:
        return command
    groups = arguments(names)
    return shlex.join(
        [
            str(PYTHON),
            "-m",
            "experiments.family_cli",
            ",".join(groups),
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
    values = parse(arguments(names), sys.argv[2:separator])
    os.environ["SPS_FAMILY_ARGUMENTS"] = json.dumps(values, separators=(",", ":"))
    command = sys.argv[separator + 1 :]
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
