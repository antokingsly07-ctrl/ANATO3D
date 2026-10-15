"""Segmentation orchestration: pick an engine, run it, record the result."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ...core.errors import SegmentationError
from ...core.logging_ import get_logger
from ...core.storage import new_id
from ...models import SegmentationMask, SegmentationResult, Volume
from .base import SegmentationEngine
from .refine import mask_statistics
from .registry import get_engine

logger = get_logger("segmentation.pipeline")

VALID_COLOR = "#8ab4f8"


def normalize_engine_key(requested: str | None, volume: Volume) -> tuple[str, list[str]]:
    """Resolve the requested engine, degrading gracefully when AI is unavailable."""
    notes: list[str] = []
    key = (requested or "threshold").strip().lower()

    if key in ("auto", "default"):
        key = "threshold" if volume.intensity_units == "HU" else "threshold"
        notes.append(f"Engine 'auto' resolved to '{key}'.")

    if key == "ai":
        engine = get_engine("ai")
        availability = engine.availability(volume)
        if not availability.available:
            notes.append(
                "AI engine requested but unavailable: "
                + (availability.reason or "no model configured.")
                + " Falling back to the classical Hounsfield threshold engine. "
                "The result is a classical segmentation, not a model prediction."
            )
            key = "classical_fallback"
        else:
            notes.append("AI engine requested and available.")

    return key, notes


def run_segmentation(
    volume: Volume,
    engine_key: str,
    params: dict[str, Any],
    *,
    progress=None,
    label: str | None = None,
    color: str | None = None,
) -> SegmentationResult:
    started = time.time()
    resolved_key, notes = normalize_engine_key(engine_key, volume)
    engine: SegmentationEngine = get_engine(resolved_key)

    availability = engine.availability(volume)
    if not availability.available:
        raise SegmentationError(
            f"Segmentation engine '{engine.display_name}' is not available in this deployment.",
            detail=availability.reason or "",
        )

    run_params = dict(params or {})
    if label:
        run_params["label"] = label
    if color:
        run_params["color"] = color

    if progress:
        progress.step("segmentation", 0.04, f"Running '{engine.display_name}'")

    mask: SegmentationMask = engine.run(volume, run_params, progress=progress)

    if progress:
        progress.step("segmentation", 0.45, "Segmentation complete; computing mask statistics")

    stats = mask_statistics(mask.mask, volume.voxel_volume_mm3)
    if stats["voxel_count"] == 0:
        raise SegmentationError("Segmentation produced an empty mask.")

    segmentation_id = new_id("seg")
    result = SegmentationResult(
        segmentation_id=segmentation_id,
        volume_id=volume.volume_id,
        label=mask.label,
        color=color or mask.color or VALID_COLOR,
        engine=mask.engine,
        params={
            **mask.params,
            "requested_engine": engine_key,
            "engine_notes": notes,
            "statistics": stats,
        },
        mask=np.asarray(mask.mask, dtype=bool),
        created_at=time.time(),
        notes=list(mask.notes) + notes,
        warnings=list(mask.warnings),
        duration_s=time.time() - started,
    )
    if progress:
        progress.step(
            "segmentation",
            0.5,
            f"Mask: {stats['voxel_count']} voxels ({stats['volume_cm3']} cm3) in "
            f"{stats['components']} component(s)",
        )
    return result
