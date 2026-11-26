"""Reserve a configured NFS destination before submitting its MLSpace worker."""

import json
import fcntl
import shutil
from contextlib import contextmanager
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
        if not relative.parts or relative.parts[0] in {
            "in_progress",
            "failed",
            ".launch-locks",
        }:
            raise ValueError(f"Invalid result destination {self.output}")

    @property
    def reservation(self):
        relative = self.output.resolve().relative_to(self.root.resolve())
        return self.root / "in_progress" / relative.parent / (relative.name + ".active")

    @property
    def work(self):
        return self.reservation / "result"

    @property
    def failed_attempt(self):
        relative = self.output.resolve().relative_to(self.root.resolve())
        return self.root / "failed" / relative.parent / (relative.name + ".last")

    @contextmanager
    def submitting(self):
        """Serialize receipt reconciliation. The persistent claim is the atomic mkdir."""
        if self.output.is_dir():
            yield False
            return
        relative = self.reservation.relative_to(self.root / "in_progress")
        lock = self.root / ".launch-locks" / relative
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            yield not self.output.is_dir()

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
        previous = self.failed_attempt / "result"
        if previous.is_dir():
            previous.rename(self.work)
        return True

    @contextmanager
    def writing(self, reserved):
        if not reserved and not self.reserve():
            yield None
            return
        if not self.reservation.is_dir():
            raise RuntimeError(f"Missing submission reservation {self.reservation}")
        try:
            self.work.mkdir(parents=True, exist_ok=True)
            yield self.work
            self.finish()
        except Exception:
            if not reserved and self.reservation.is_dir():
                self.retire()
            raise

    def finish(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.work.rename(self.output)
        shutil.rmtree(self.reservation)

    def retire(self):
        # Preserve partial checkpoints, logs and the original receipt for diagnosis.
        archive = self.failed_attempt
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            archive.rename(archive.with_name(uuid4().hex))
        self.reservation.rename(archive)

    def record(self, job_name):
        path = self.reservation / "job.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"job_name": job_name}))
        tmp.replace(path)

    def job_name(self):
        path = self.reservation / "job.json"
        return json.loads(path.read_text())["job_name"] if path.is_file() else None


@contextmanager
def preparation_lock(root):
    """Serialize data creation and probes that share a family's working files."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".preparation.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield
