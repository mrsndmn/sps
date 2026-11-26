"""Select a named plan before loading any suite-specific worker modules."""

import argparse
import os
import subprocess
import sys

from orchestration.named.registry import FAMILIES, named_runs, resolve
from orchestration.named import special_families


def standalone_plans(family):
    os.environ.update(special_families.environment(family))
    plans = {}
    for plan in special_families.FAMILIES[family]():
        key = (plan.name, plan.phase)
        if key in plans:
            raise ValueError(f"Duplicate plan {family}/{key}")
        plans[key] = plan
    return plans


def main(default_phase="train"):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--experiment", metavar="FAMILY/NAME")
    select.add_argument("--list", nargs="?", const="", metavar="FAMILY")
    args, remaining = parser.parse_known_args()
    if args.list is not None:
        if not args.list:
            print("\n".join(sorted(FAMILIES.keys() | special_families.FAMILIES.keys())))
        elif args.list in special_families.FAMILIES:
            for name, phase in standalone_plans(args.list):
                print(f"{args.list}/{name}\t{phase}")
        elif args.list in FAMILIES:
            for run in FAMILIES[args.list]():
                print(f"{args.list}/{run.name}")
        else:
            parser.error(f"Unknown family {args.list!r}")
        return 0
    if not args.experiment:
        from orchestration.mlspace_launch import main as launch

        return launch(default_phase)
    family, _, name = args.experiment.partition("/")
    if family in special_families.FAMILIES:
        selection = argparse.ArgumentParser(
            description="Launch one specialized experiment"
        )
        selection.add_argument("--phase", default=default_phase)
        selection.add_argument("--dry", action="store_true")
        options = selection.parse_args(remaining)
        plans = standalone_plans(family)
        plan = plans.get((name, options.phase))
        if plan is None:
            selection.error(f"Unknown run or phase. See --list {family}")
        if options.dry:
            from orchestration.named.preview import preview

            return preview(plan)
        cmd = [sys.executable, plan.script, *plan.args]
        if plan.submit_flag:
            cmd.append(plan.submit_flag)
        env = dict(
            os.environ,
            SPS_FAMILY_SCHEMAS=",".join(plan.schemas),
            SPS_SELECTED_RUN=plan.selected_run or plan.name,
        )
        if plan.output_dir is None:
            return subprocess.call(cmd, env=env)
        from orchestration.named.publication import submitting

        with submitting(plan.output_dir) as ready:
            if ready:
                env["SPS_SUBMIT_OUTPUT"] = str(plan.output_dir)
                return subprocess.call(cmd, env=env)
        return 0
    try:
        family, run = resolve(args.experiment, named_runs())
    except ValueError as error:
        parser.error(str(error))
    if any(arg.split("=")[0] == "--filter" for arg in remaining):
        parser.error("--experiment already selects one run")
    os.environ["SPS_EXPERIMENT_SUITE"] = family
    sys.argv[1:] = [*remaining, "--filter", run.name]
    from orchestration.mlspace_launch import main as launch

    return launch(default_phase, named=run.name)
