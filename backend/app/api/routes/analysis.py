"""Measurement and analysis endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from .. import deps

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.get("/{analysis_id}", summary="Retrieve measurements for a generated model")
def get_analysis(analysis_id: str) -> dict:
    """Return the measurement/analysis payload produced when a mesh was built.

    ``analysis_id`` may be either a mesh id or a segmentation id; both are
    registered when a mesh is generated.
    """
    return deps.get_analysis(analysis_id).payload


__all__ = ["router"]
