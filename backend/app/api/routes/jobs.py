"""Background job status endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ...schemas.common import JobStatusResponse
from .. import deps

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", summary="List recent jobs")
def list_jobs(limit: int = 25) -> list[dict]:
    jobs = deps.job_manager.list(limit=limit)
    return [deps.status_envelope(job).model_dump(mode="json") for job in jobs]


@router.get("/{job_id}", response_model=JobStatusResponse, summary="Poll a job")
def get_job(job_id: str) -> JobStatusResponse:
    """Poll a job submitted with ``wait=false``.

    ``result`` is populated only once ``status`` is ``completed``.
    """
    return deps.status_envelope(deps.get_job(job_id))


__all__ = ["router"]
