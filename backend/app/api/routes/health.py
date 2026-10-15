"""Health and capability reporting."""

from __future__ import annotations

from fastapi import APIRouter

from ...core.storage import store
from ...schemas.common import HealthResponse
from ...services.segmentation.registry import get_engine
from ..deps import settings

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Service health and capabilities")
def health() -> HealthResponse:
    """Report liveness, configured limits and AI availability.

    The response never contains patient data.  ``ai_segmentation_available``
    reflects the real state of the optional MONAI adapter -- it is ``False``
    unless a model bundle has been configured and MONAI/torch are importable.
    """
    availability = get_engine("ai").availability(None)  # type: ignore[arg-type]

    return HealthResponse(
        status="ok",
        application=settings.app_name,
        version=settings.app_version,
        ai_segmentation_available=bool(availability.available),
        ai_unavailable_reason=availability.reason,
        limits={
            "max_file_bytes": settings.max_file_bytes,
            "max_upload_bytes": settings.max_upload_bytes,
            "max_upload_files": settings.max_upload_files,
            "max_voxels_soft": settings.max_voxels_soft,
            "max_voxels_hard": settings.max_voxels_hard,
            "workspace_ttl_minutes": settings.workspace_ttl_minutes,
        },
        storage={
            "data_dir": str(settings.work_root),
            **store.stats(),
        },
    )


__all__ = ["router"]
