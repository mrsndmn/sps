"""Submit the existing cap-study evaluation workers with one GPU."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
from unittest.mock import patch

from scripts import mlspace_launch as launch
from scripts import qwen35_cap_compare as compare
from scripts import teacher_cap_probe as teacher
from scripts.source_compatibility import source_hash

PUBLISHED_TEACHER_COMMIT = "7fad856aa3c8d3bcd11c08b37f1f444e7601f2e8"


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
        worker = ["scripts.qwen35_cap_compare", "worker", "--arm", args.arm]
    else:
        root = teacher.ROOT
        worker = ["scripts.teacher_cap_probe"]
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
        body.update(
            job_desc=f"sps-{root.name} eval {name} {identity} #d.tarasov @mrsndmn",
            script=f"cd {shlex.quote(str(staged))} && "
            + shlex.join([str(launch.PYTHON), "-m", *worker]),
        )
        return body

    def report_path():
        folder = (
            compare.folder(args.arm)
            if args.arm
            else root / source_hash(teacher.__file__)
        )
        return folder / "report.json"

    if args.dry:
        marker = None if args.arm and not compare.PROMPTS.is_file() else report_path()
        print(json.dumps(payload(marker), indent=2))
        return
    from mls.manager.job.utils import (
        get_in_progress_jobs,
        training_job_api_from_profile,
    )

    with patch.object(launch, "ARTIFACTS_ROOT", root), launch.submission_lock():
        if args.arm:
            compare.prepare()
        marker = report_path()
        body = payload(marker)
        if marker.is_file():
            print(f"Completed {name}: {marker}")
            return
        client, options = training_job_api_from_profile("default")
        launch.disable_submission_retries(client)
        active = {
            launch.normalize(j["job_desc"]): j["job_name"]
            for j in get_in_progress_jobs(client_profile="default")
        }
        identity = launch.normalize(body["job_desc"])
        published_teacher = not args.arm and marker.parent.name == teacher.PILOT_HASH
        # Historical cap workers write report.json as their success marker.
        # Check the API for receipts whose jobs are not in the active list.
        for receipt in (root / "submissions").glob("*.json"):
            row = json.loads(receipt.read_text())
            is_published_teacher = (
                published_teacher
                and row["identity"] == "qwen3-teacher-cap-probe"
                and row.get("commit") == PUBLISHED_TEACHER_COMMIT
            )
            if row["identity"] != identity and not is_published_teacher:
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
        launch.reconcile_intents(active)
        if identity in active:
            print(active[identity])
            return
        launch.stage_repo(Path(launch.git("rev-parse", "--show-toplevel")), commit)
        body["region"] = options["region"]
        print(json.dumps(launch.submit_payload(client, body, "eval", name, commit)))


if __name__ == "__main__":
    main()
