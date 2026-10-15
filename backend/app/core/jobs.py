"""Progress reporting + background job management.

Heavy operations (volume reconstruction, segmentation, mesh generation) run in a
thread pool so the HTTP event loop is never blocked.  Progress is *real*: the
service functions call :meth:`ProgressReporter.step` at genuine checkpoints and
the numbers come from actual loop counters, never from a timer.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from ..core.errors import AnatoError
from ..core.logging_ import get_logger

logger = get_logger("jobs")


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class LogEntry:
    time: float
    level: str
    message: str
    stage: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "level": self.level,
            "message": self.message,
            "stage": self.stage,
        }


@dataclass
class Job:
    job_id: str
    kind: str
    status: JobStatus = JobStatus.PENDING
    stage: str = "queued"
    progress: float = 0.0
    message: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: Any = None
    error: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    logs: list[LogEntry] = field(default_factory=list)
    cancel_requested: bool = False

    def add_log(self, level: str, message: str, stage: str | None = None) -> None:
        self.logs.append(LogEntry(time=time.time(), level=level, message=message, stage=stage))

    def as_dict(self, *, include_logs: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status.value,
            "stage": self.stage,
            "progress": round(float(self.progress), 4),
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": (
                round((self.finished_at or time.time()) - self.started_at, 3)
                if self.started_at
                else None
            ),
            "warnings": list(self.warnings),
            "error": self.error,
            "result": self.result,
            "result_ready": self.result is not None,
        }
        if include_logs:
            payload["logs"] = [entry.as_dict() for entry in self.logs]
        return payload


#: Canonical processing stages, in execution order.  The UI renders these as a
#: checklist and marks each one as it is actually reached.
PIPELINE_STAGES: tuple[str, ...] = (
    "dicom_loading",
    "metadata_analysis",
    "slice_ordering",
    "volume_reconstruction",
    "preprocessing",
    "segmentation",
    "mesh_generation",
    "mesh_refinement",
    "validation",
)


class ProgressReporter:
    """Thread-safe progress sink handed to service functions."""

    def __init__(self, job: Job | None = None, *, on_log: Callable[[str, str], None] | None = None):
        self._job = job
        self._on_log = on_log
        self._lock = threading.Lock()

    # -- logging -----------------------------------------------------------
    def log(self, message: str, level: str = "info", stage: str | None = None) -> None:
        with self._lock:
            if self._job is not None:
                self._job.add_log(level, message, stage)
            if self._on_log is not None:
                self._on_log(level, message)
        getattr(logger, level if level in ("info", "warning", "error", "debug") else "info")(
            message
        )

    # -- progress ----------------------------------------------------------
    def step(self, stage: str, fraction: float, message: str = "") -> None:
        frac = max(0.0, min(1.0, float(fraction)))
        with self._lock:
            if self._job is not None:
                self._job.stage = stage
                self._job.progress = frac
                if message:
                    self._job.message = message
                if message:
                    self._job.add_log("info", message, stage)
            if message:
                logger.info("[%s %.0f%%] %s", stage, frac * 100, message)

    def warn(self, message: str, stage: str | None = None) -> None:
        with self._lock:
            if self._job is not None:
                self._job.warnings.append(message)
        self.log(message, level="warning", stage=stage)


class JobManager:
    def __init__(self, *, max_workers: int = 4) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="anato3d")

    # -- lifecycle ---------------------------------------------------------
    def create(self, kind: str) -> Job:
        job = Job(job_id=f"job_{uuid.uuid4().hex[:16]}", kind=kind)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, *, limit: int = 50) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def submit(self, job: Job, fn: Callable[[ProgressReporter], Any]) -> Job:
        """Run ``fn`` in the thread pool, updating ``job`` as it goes."""

        def runner() -> None:
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            reporter = ProgressReporter(job)
            try:
                result = fn(reporter)
                job.result = result
                job.status = JobStatus.COMPLETED
                job.progress = 1.0
                job.stage = "done"
                job.message = job.message or "Completed"
            except AnatoError as exc:
                job.status = JobStatus.FAILED
                job.error = exc.to_dict()
                job.message = exc.message
                job.add_log("error", exc.message)
                logger.warning("Job %s failed: %s", job.job_id, exc.message)
            except Exception as exc:  # pragma: no cover - defensive
                job.status = JobStatus.FAILED
                job.error = {
                    "code": "internal_error",
                    "message": "An unexpected error occurred during processing.",
                }
                job.message = "An unexpected error occurred during processing."
                job.add_log("error", f"Unexpected error: {type(exc).__name__}")
                logger.exception("Job %s crashed", job.job_id)
            finally:
                job.finished_at = time.time()

        self._pool.submit(runner)
        return job

    def run_sync(self, kind: str, fn: Callable[[ProgressReporter], Any]) -> Job:
        """Run ``fn`` inline (used by tests and small requests)."""
        job = self.create(kind)
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        reporter = ProgressReporter(job)
        try:
            job.result = fn(reporter)
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.stage = "done"
        except AnatoError as exc:
            job.status = JobStatus.FAILED
            job.error = exc.to_dict()
            job.message = exc.message
        except Exception:  # pragma: no cover - defensive
            logger.exception("Synchronous job %s crashed", job.job_id)
            job.status = JobStatus.FAILED
            job.error = {"code": "internal_error", "message": "Unexpected internal error."}
        finally:
            job.finished_at = time.time()
        return job

    def prune(self, *, max_jobs: int = 200, max_age_s: float = 3600.0) -> int:
        now = time.time()
        removed = 0
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    if now - job.created_at > max_age_s:
                        del self._jobs[job_id]
                        removed += 1
            if len(self._jobs) > max_jobs:
                finished = sorted(
                    (j for j in self._jobs.values() if j.status in (JobStatus.COMPLETED, JobStatus.FAILED)),
                    key=lambda j: j.created_at,
                )
                for job in finished[: len(self._jobs) - max_jobs]:
                    self._jobs.pop(job.job_id, None)
                    removed += 1
        return removed
