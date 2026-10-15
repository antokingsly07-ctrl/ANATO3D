"""Segmentation endpoints.

Engines and presets are volume-aware: availability and sensible defaults
depend on the intensity domain and geometric properties of the reconstructed
volume.
"""

from __future__ import annotations

import base64

import numpy as np
from fastapi import APIRouter, Query

from ...core.errors import ValidationError
from ...core.logging_ import get_logger
from ...core.storage import store
from ...models.records import SegmentationRecord
from ...schemas.common import JobEnvelope
from ...schemas.processing import SegmentationRequest
from ...services.reconstruction.slices import extract_mask_plane
from ...services.segmentation import pipeline, registry
from ...services.validation import checks as checks_module
from .. import deps
from .. import presenters

logger = get_logger("api.segmentation")

router = APIRouter(prefix="/segmentation", tags=["segmentation"])


@router.get("/engines", summary="List available segmentation engines")
def engines(volume_id: str | None = Query(None)) -> list[dict]:
    volume = deps.find_volume(volume_id).volume if volume_id else None
    return registry.list_engines(volume)


@router.get("/presets", summary="List anatomical segmentation presets")
def presets(
    volume_id: str | None = Query(None),
    modality: str | None = Query(None),
) -> list[dict]:
    resolved = modality
    if resolved is None and volume_id:
        record = deps.find_volume(volume_id)
        if record is not None:
            resolved = record.volume.modality
    return registry.list_presets(resolved)


_PRESET_BY_KEY = {preset["key"]: preset for preset in registry.PRESETS}


def _resolve_segmentation(request: SegmentationRequest) -> tuple[str, dict, str | None, str | None]:
    """Merge a preset with the request, honouring the documented precedence."""
    preset = None
    if request.preset:
        preset = _PRESET_BY_KEY.get(request.preset)
        if preset is None:
            raise ValidationError(
                f"Unknown segmentation preset '{request.preset}'.",
                detail=f"Available presets: {', '.join(sorted(_PRESET_BY_KEY))}.",
            )

    params: dict = dict(preset["params"]) if preset else {}
    params.update(request.params or {})

    if request.engine:
        engine = request.engine
    elif preset:
        engine = preset["engine"]
    else:
        engine = "threshold"

    label = request.label or (preset["params"].get("label") if preset else None)
    color = request.color or (preset.get("color") if preset else None)
    return engine, params, label, color


@router.post("/run", response_model=JobEnvelope, summary="Run segmentation")
def run(
    request: SegmentationRequest,
    wait: bool = Query(True, description="Run inline and return the result."),
) -> JobEnvelope:
    volume_record = deps.get_volume(request.volume_id)
    volume = volume_record.volume
    dataset_id = volume_record.dataset_id
    engine_key, params, label, color = _resolve_segmentation(request)

    def job(reporter):
        reporter.step("segmentation", 0.01, "Resolving engine and parameters")
        result = pipeline.run_segmentation(
            volume,
            engine_key,
            params,
            progress=reporter,
            label=label,
            color=color,
        )
        store.save_mask(dataset_id, result.segmentation_id, result.mask)
        findings = []
        try:
            findings.extend([f.as_dict() for f in checks_module.check_segmentation(result, volume)])
        except Exception:  # noqa: BLE001 - checks must not abort the job
            logger.exception("check_segmentation failed")

        record = SegmentationRecord(
            dataset_id=dataset_id,
            result=result,
            findings=findings,
        )
        deps.register_segmentation(record)
        reporter.step("done", 1.0, "Segmentation complete")
        return presenters.segmentation_response(record)

    return deps.dispatch("segmentation", job, wait=wait)


@router.get("/{segmentation_id}", summary="Retrieve a segmentation")
def get_segmentation(segmentation_id: str) -> dict:
    record = deps.get_segmentation(segmentation_id)
    return presenters.segmentation_response(record)


@router.get("/{segmentation_id}/mask", summary="Extract a mask slice")
def get_mask_slice(
    segmentation_id: str,
    plane: str = Query("axial", pattern="^(axial|coronal|sagittal)$"),
    index: int = Query(0, ge=0),
) -> dict:
    record = deps.get_segmentation(segmentation_id)
    volume_record = deps.get_volume(record.result.volume_id)
    result = record.result
    if result.mask is None:  # pragma: no cover
        raise ValidationError("Mask is not available.")

    from ...services.reconstruction import slices as slices_mod

    counts = slices_mod.plane_counts(volume_record.volume)
    if counts.get(plane, 0) == 0:
        raise ValidationError(f"No slices along the {plane} plane for this volume.")

    resolved = min(index, counts[plane] - 1) if counts[plane] > 0 else index
    plane_slice = extract_mask_plane(result.mask, volume_record.volume, plane, resolved)
    packed = np.packbits(np.ascontiguousarray(plane_slice))
    encoded = base64.b64encode(packed.tobytes()).decode("ascii")

    return {
        "segmentation_id": segmentation_id,
        "plane": plane,
        "index": resolved,
        "rows": int(plane_slice.shape[0]),
        "columns": int(plane_slice.shape[1]),
        "encoding": "packbits-base64",
        "data": encoded,
        "shape": [int(plane_slice.shape[0]), int(plane_slice.shape[1])],
    }


@router.delete("/{segmentation_id}", summary="Drop a segmentation")
def delete_segmentation(segmentation_id: str):
    store.drop("segmentations", segmentation_id)
    return {"segmentation_id": segmentation_id, "status": "dropped"}


__all__ = ["router"]
