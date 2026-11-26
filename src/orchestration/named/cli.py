"""Select an exact family/run and forward it to the family's native launcher."""

import argparse
import os
import subprocess
import sys

from orchestration.named.registry import FAMILIES, family_module, family_runs
from orchestration.suite_registry import INSTRUCT_SUITE


def main(default_phase="train"):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--list", nargs="?", const="", metavar="FAMILY")
    select.add_argument("--experiment", metavar="FAMILY/NAME")
    args, remaining = parser.parse_known_args()
    if args.list is not None:
        if remaining:
            parser.error("--list accepts only a family name")
        if not args.list:
            print("\n".join(sorted(FAMILIES)))
        elif args.list in FAMILIES:
            for name, phase in family_runs(args.list):
                print(f"{args.list}/{name}\t{phase}")
        else:
            parser.error(f"Unknown family {args.list!r}")
        return 0
    if not args.experiment:
        from orchestration.mlspace_launch import main as launch

        return launch(default_phase)

    family, _, name = args.experiment.partition("/")
    if family not in FAMILIES or not name:
        parser.error("Use --experiment FAMILY/NAME. See --list.")
    phase_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    phase_parser.add_argument("--phase", default=default_phase)
    phase, _ = phase_parser.parse_known_args(remaining)
    run = family_runs(family).get((name, phase.phase))
    if run is None:
        parser.error(
            f"Unknown {phase.phase} run {args.experiment!r}. See --list {family}"
        )
    selection = argparse.ArgumentParser(parents=[phase_parser], allow_abbrev=False)
    selection.add_argument("--dry", action="store_true")
    for key in run.options:
        selection.add_argument("--" + key, required=key in run.required)
    opts = selection.parse_args(remaining)
    cmd = [sys.executable, run.script, *run.args]
    for key in run.options:
        value = getattr(opts, key)
        if value is not None:
            cmd.extend(("--" + key, value))
    if opts.dry:
        cmd.append("--dry")
    env = dict(os.environ, SPS_EXPERIMENT_SUITE=INSTRUCT_SUITE)
    configure = getattr(family_module(family), "launch_environment", None)
    if configure:
        env.update(configure(family))
    return subprocess.call(cmd, env=env)
