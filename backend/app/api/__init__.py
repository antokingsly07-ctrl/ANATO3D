"""ANATO3D HTTP API package."""

from __future__ import annotations

from fastapi import APIRouter

from .routes import (
    analysis,
    dicom,
    export,
    health,
    jobs,
    mesh,
    reconstruction,
    segmentation,
    validation,
)

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(dicom.router)
api_router.include_router(reconstruction.router)
api_router.include_router(segmentation.router)
api_router.include_router(mesh.router)
api_router.include_router(validation.router)
api_router.include_router(analysis.router)
api_router.include_router(export.router)
api_router.include_router(jobs.router)

__all__ = ["api_router"]
