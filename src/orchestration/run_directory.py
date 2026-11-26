"""Reserve a configured NFS destination before submitting its MLSpace worker."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class RunDirectory:
    root: Path
    output: Path

    def __post_init__(self):
        if not self.root.is_absolute() or not self.output.is_absolute():
            raise ValueError("Artifact paths must be absolute")
        relative = self.output.resolve().relative_to(self.root.resolve())
        if not relative.parts or relative.parts[0] in {"in_progress", "failed"}:
            raise ValueError(f"Invalid result destination {self.output}")

    @property
    def reservation(self):
        relative = self.output.resolve().relative_to(self.root.resolve())
        return self.root / "in_progress" / relative.parent / (relative.name + ".active")

    @property
    def work(self):
        return self.reservation / "result"

    def reserve(self):
        if self.output.exists():
            return False
        self.reservation.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.reservation.mkdir()
        except FileExistsError:
            return False
        if self.output.exists():
            self.reservation.rmdir()
            return False
        return True

    def finish(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.work.rename(self.output)
        shutil.rmtree(self.reservation)

    def retire(self):
        # Preserve partial checkpoints, logs and the original receipt for diagnosis.
        archive = self.root / "failed" / uuid4().hex
        archive.parent.mkdir(parents=True, exist_ok=True)
        self.reservation.rename(archive)

    def record(self, job_name):
        path = self.reservation / "job.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"job_name": job_name}))
        tmp.replace(path)

    def job_name(self):
        path = self.reservation / "job.json"
        return json.loads(path.read_text())["job_name"] if path.is_file() else None
