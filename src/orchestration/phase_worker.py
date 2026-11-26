"""Publish one legacy worker's phase using the named result directory."""

import argparse
import os
import subprocess

from orchestration import experiment_suite as suite
from orchestration.pilot_launch import destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "train", "eval"))
    parser.add_argument("name")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    run = None
    if args.phase != "prepare":
        (run,) = [r for r in suite.collect_experiments() if r.name == args.name]
    directory = destination(args.phase, run)
    with directory.writing(reserved=True) as work:
        env = dict(
            os.environ,
            SPS_WORK_PHASE=args.phase,
            SPS_PHASE_RUN=args.name,
            SPS_PHASE_WORK=str(work),
            SPS_PUBLISHED_OUTPUT=str(directory.output),
        )
        subprocess.run(args.command, env=env, check=True)


if __name__ == "__main__":
    main()
