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
from orchestration.source_compatibility import source_hash
from experiments import qwen35_cap_compare as comparison_config
from experiments import teacher_cap_probe as teacher_config

PUBLISHED_TEACHER_COMMIT = "7fad856aa3c8d3bcd11c08b37f1f444e7601f2e8"
PUBLISHED_COMPARISON_COMMIT = "4e626e62f8af700b2e04c9ad2ab7e502a5e3a81a"
PUBLISHED_COMPARISON_DESIGN = (
    "368b7c39020d6f4dda4a336ec92f8dc221e24ca187bf6539d9aec3dea2380e12",
    "185b483f6ef8312477e2f7895f7d879835cb129c6332c4aa3966d44317042406",
)


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
        published_receipt = None
        if (
            args.arm
            and marker.relative_to(root).parts[:2] == PUBLISHED_COMPARISON_DESIGN
        ):
            published_receipt = (f"qwen35-cap-{args.arm}", PUBLISHED_COMPARISON_COMMIT)
        elif not args.arm and marker.parent.name == teacher.PILOT_HASH:
            published_receipt = ("qwen3-teacher-cap-probe", PUBLISHED_TEACHER_COMMIT)
        # Historical cap workers write report.json as their success marker.
        # Check the API for receipts whose jobs are not in the active list.
        for receipt in (root / "submissions").glob("*.json"):
            row = json.loads(receipt.read_text())
            historical_identity = (row["identity"], row.get("commit"))
            if row["identity"] != identity and historical_identity != published_receipt:
                continue
            state = str(client.get_job_status(row["job_name"]).get("status")).casefold()
            if state not in {
                "failed",
                "cancelled",
                "canceled",
                "terminated",
                "stopped",
                "deleted",
            }:
                if state in {"completed", "succeeded"}:
                    raise RuntimeError(f"Completed job missing {marker}")
                active[row["identity"]] = row["job_name"]
                active.setdefault(identity, row["job_name"])
        launch.reconcile_intents(active, {marker.parent})
        if identity in active:
            print(active[identity])
            return
        launch.stage_repo(Path(launch.git("rev-parse", "--show-toplevel")), commit)
        body["region"] = options["region"]
        print(json.dumps(launch.submit_payload(client, body, "eval", name, commit, output_dir=marker.parent)))
