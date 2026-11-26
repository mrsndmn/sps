"""Submit the existing cap-study evaluation workers with one GPU."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
from unittest.mock import patch

from orchestration import mlspace_launch as launch
from workloads.evaluation import qwen35_cap_compare as compare
from workloads.evaluation import teacher_cap_probe as teacher
from experiments import qwen35_cap_compare as comparison_config
from experiments import teacher_cap_probe as teacher_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study", choices=("teacher", "compare"))
    parser.add_argument("arm", nargs="?", choices=tuple(compare.experiment_configs))
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    if (args.study == "compare") != bool(args.arm):
        parser.error("Select an arm only for the paired comparison")
    if args.arm:
        root = compare.ROOT
        worker = ["scripts/evaluation/qwen35_cap_compare.py", "worker", "--arm", args.arm]
    else:
        root = teacher.ROOT
        worker = ["scripts/evaluation/teacher_cap_probe.py"]
    name = args.arm or "continuations"
    commit = launch.git("rev-parse", "HEAD")
    staged = launch.STAGING_ROOT / commit

    def payload(marker):
        if marker is None:
            identity = "cohort-pending"
        else:
            # The worker chooses a fresh output directory for a changed design
            # or paired cohort. Receipts and active jobs use the same identity.
            relative = marker.relative_to(root).as_posix()
            identity = hashlib.sha256(relative.encode()).hexdigest()
        body = launch.payload("eval", name, staged, "d.tarasov", "mrsndmn")
        config = comparison_config if args.arm else teacher_config
        if args.arm:
            body["env_variables"].update(comparison_config.RUNTIME_ENV)
        body.update(
            job_desc=f"sps-{root.name} eval {name} {identity} #d.tarasov @mrsndmn",
            script=f"cd {shlex.quote(str(staged))} && "
            + shlex.join(
                ["timeout", f"{config.EVAL_HOURS}h", str(launch.PYTHON), *worker]
            ),
        )
        return body

    def report_path():
        folder = (
            compare.folder(args.arm)
            if args.arm
            else root / teacher_config.OUTPUT_NAME
        )
        return folder / "report.json"

    if args.dry:
        print(json.dumps(payload(report_path()), indent=2))
        return
    with patch.object(launch, "ARTIFACTS_ROOT", root), launch.submission_lock():
        marker = report_path()
        if marker is not None and marker.is_file():
            print(f"Completed {name}: {marker}")
            return
        if not launch.pending(marker.parent):
            return
        if args.arm:
            compare.prepare()
            marker = report_path()
            if marker is None:
                raise RuntimeError("Comparison preparation did not publish the cohort")
            if marker.is_file():
                print(f"Completed {name}: {marker}")
                return

        from mls.manager.job.utils import (
            get_in_progress_jobs,
            training_job_api_from_profile,
        )

        body = payload(marker)
        client, options = training_job_api_from_profile("default")
        launch.disable_submission_retries(client)
        active = {
            launch.normalize(j["job_desc"]): j["job_name"]
            for j in get_in_progress_jobs(client_profile="default")
        }
        identity = launch.output_identity(body["job_desc"], marker.parent, active)
        launch.reconcile_for_outputs(client, active, [marker.parent])
        if identity in active:
            print(active[identity])
            return
        launch.stage_repo(Path(launch.git("rev-parse", "--show-toplevel")), commit)
        body["region"] = options["region"]
        print(json.dumps(launch.submit_payload(client, body, "eval", name, commit, output_dir=marker.parent)))
