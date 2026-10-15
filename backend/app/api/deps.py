"""Shared API dependencies.

This module owns:

* the process-wide :class:`JobManager`,
* helpers that turn a :class:`Job` into a response envelope,
* resolvers that fetch stored artefacts or raise structured errors,
* small utilities shared by the routers.

Routers stay thin: they validate their input, resolve the artefacts they need
*eagerly* (so that bad ids produce a real HTTP error) and then hand the heavy
work to :func:`dispatch`.
"""

from __future__ import annotations

from typing import Any, Callable

from ..config import get_settings
from ..core.errors import AnatoError, NotFoundError, ValidationError
from ..core.jobs import Job, JobManager, ProgressReporter
from ..core.logging_ import get_logger
from ..core.storage import store
from ..models.records import (
    AnalysisRecord,
    DatasetRecord,
    GroundTruthRecord,
    MeshRecord,
    SegmentationRecord,
    ValidationRecord,
    VolumeRecord,
)
from ..schemas.common import JobEnvelope, JobStatusResponse

logger = get_logger("api")

settings = get_settings()

#: Background worker pool for reconstruction / segmentation / mesh / validation.
job_manager = JobManager(max_workers=4)

DISCLAIMER = (
    "This application is a research prototype and is NOT a certified medical device. "
    "It has not been cleared or approved by any regulator and must not be used for "
    "diagnosis or treatment decisions."
)

#: Buckets that the generic :func:`get_*` resolvers read from.
_BUCKETS = {
    "dataset": "datasets",
    "volume": "volumes",
    "segmentation": "segmentations",
    "mesh": "meshes",
    "validation": "validations",
    "analysis": "analyses",
    "ground_truth": "ground_truth",
}


# ---------------------------------------------------------------------------
# Job helpers
# ---------------------------------------------------------------------------

def envelope(job: Job) -> JobEnvelope:
    """Convert a job into the envelope returned by every heavy endpoint."""
    return JobEnvelope.model_validate(job.as_dict())


def status_envelope(job: Job) -> JobStatusResponse:
    return JobStatusResponse.model_validate(job.as_dict())


def dispatch(kind: str, fn: Callable[[ProgressReporter], Any], *, wait: bool) -> JobEnvelope:
    """Run ``fn`` inline (``wait=True``) or in the worker pool, then envelope it."""
    if wait:
        job = job_manager.run_sync(kind, fn)
    else:
        job = job_manager.create(kind)
        job_manager.submit(job, fn)
    return envelope(job)


def get_job(job_id: str) -> Job:
    job = job_manager.get(job_id)
    if job is None:
        raise NotFoundError(
            "No processing job with that id exists. It may have expired; start the operation again."
        )
    return job


# ---------------------------------------------------------------------------
# Artefact resolvers
# ---------------------------------------------------------------------------

def _resolve(bucket: str, key: str, message: str) -> Any:
    record = store.find(bucket, key)
    if record is None:
        raise NotFoundError(message)
    return record


def get_dataset(dataset_id: str) -> DatasetRecord:
    return _resolve(
        _BUCKETS["dataset"],
        dataset_id,
        "That dataset is not available. It may have expired -- upload the DICOM files again.",
    )


def find_dataset(dataset_id: str) -> DatasetRecord | None:
    return store.find(_BUCKETS["dataset"], dataset_id)


def get_volume(volume_id: str) -> VolumeRecord:
    return _resolve(
        _BUCKETS["volume"],
        volume_id,
        "That reconstructed volume is not available. Run the reconstruction step first.",
    )


def find_volume(volume_id: str) -> VolumeRecord | None:
    return store.find(_BUCKETS["volume"], volume_id)


def get_segmentation(segmentation_id: str) -> SegmentationRecord:
    return _resolve(
        _BUCKETS["segmentation"],
        segmentation_id,
        "That segmentation is not available. Run the segmentation step first.",
    )


def find_segmentation(segmentation_id: str) -> SegmentationRecord | None:
    return store.find(_BUCKETS["segmentation"], segmentation_id)


def get_mesh(mesh_id: str) -> MeshRecord:
    return _resolve(
        _BUCKETS["mesh"],
        mesh_id,
        "That 3D model is not available. Generate the mesh first.",
    )


def find_mesh(mesh_id: str) -> MeshRecord | None:
    return store.find(_BUCKETS["mesh"], mesh_id)


def get_validation(validation_id: str) -> ValidationRecord:
    return _resolve(
        _BUCKETS["validation"],
        validation_id,
        "That validation report is not available. Run validation first.",
    )


def find_validation(validation_id: str) -> ValidationRecord | None:
    return store.find(_BUCKETS["validation"], validation_id)


def get_analysis(analysis_id: str) -> AnalysisRecord:
    return _resolve(
        _BUCKETS["analysis"],
        analysis_id,
        "That analysis is not available. Generate a 3D model first.",
    )


def find_analysis(analysis_id: str) -> AnalysisRecord | None:
    return store.find(_BUCKETS["analysis"], analysis_id)


def get_ground_truth(ground_truth_id: str) -> GroundTruthRecord:
    return _resolve(
        _BUCKETS["ground_truth"],
        ground_truth_id,
        "That reference mask is not available. Upload it again.",
    )


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------

def register_dataset(record: DatasetRecord) -> DatasetRecord:
    store.put(_BUCKETS["dataset"], record.dataset_id, record)
    return record


def register_volume(record: VolumeRecord) -> VolumeRecord:
    store.put(_BUCKETS["volume"], record.volume_id, record)
    return record


def register_segmentation(record: SegmentationRecord) -> SegmentationRecord:
    store.put(_BUCKETS["segmentation"], record.segmentation_id, record)
    return record


def register_mesh(record: MeshRecord) -> MeshRecord:
    store.put(_BUCKETS["mesh"], record.mesh_id, record)
    return record


def register_validation(record: ValidationRecord) -> ValidationRecord:
    store.put(_BUCKETS["validation"], record.validation_id, record)
    return record


def register_analysis(key: str, record: AnalysisRecord) -> AnalysisRecord:
    store.put(_BUCKETS["analysis"], key, record)
    return record


def register_ground_truth(record: GroundTruthRecord) -> GroundTruthRecord:
    store.put(_BUCKETS["ground_truth"], record.ground_truth_id, record)
    return record


# ---------------------------------------------------------------------------
# Cross-cutting lookups
# ---------------------------------------------------------------------------

def series_info_for_volume(dataset: DatasetRecord, volume: VolumeRecord):
    """Return the cached :class:`SeriesInfo` for a volume, or ``None``."""
    analysis = dataset.analysis
    if analysis is None or not volume.series_instance_uid:
        return None
    entry = analysis.series_by_uid(volume.series_instance_uid)
    return entry.info if entry is not None else None


def require_dataset_for(dataset_id: str | None) -> DatasetRecord | None:
    if not dataset_id:
        return None
    return get_dataset(dataset_id)


def validate_choice(value: str, allowed: tuple[str, ...], label: str) -> str:
    if value not in allowed:
        raise ValidationError(
            f"Unknown {label} '{value}'. Valid options: {', '.join(allowed)}.",
        )
    return value


__all__ = [
    "DISCLAIMER",
    "dispatch",
    "envelope",
    "find_analysis",
    "find_dataset",
    "find_mesh",
    "find_segmentation",
    "find_validation",
    "find_volume",
    "get_analysis",
    "get_dataset",
    "get_ground_truth",
    "get_job",
    "get_mesh",
    "get_segmentation",
    "get_validation",
    "get_volume",
    "job_manager",
    "register_analysis",
    "register_dataset",
    "register_ground_truth",
    "register_mesh",
    "register_segmentation",
    "register_validation",
    "register_volume",
    "require_dataset_for",
    "series_info_for_volume",
    "settings",
    "status_envelope",
    "validate_choice",
    "AnatoError",
]
