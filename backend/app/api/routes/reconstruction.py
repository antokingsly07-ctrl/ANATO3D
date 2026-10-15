"""Volume reconstruction endpoints.

The reconstruction is driven entirely by patient-space geometry (see
:mod:`app.services.dicom.series_ordering`).  Ordering by filename or
``InstanceNumber`` is opt-in and always flagged.
"""

from __future__ import annotations

import base64

import numpy as np
from fastapi import APIRouter, Query, Response

from ...core.errors import ValidationError
from ...core.logging_ import get_logger
from ...core.storage import store
from ...models.records import VolumeRecord
from ...schemas.common import JobEnvelope
from ...schemas.processing import (
    PreprocessRequest,
    ReconstructionRequest,
    SliceResponse,
)
from ...services.dicom import series as series_service
from ...services.reconstruction import preprocessing as preprocessing_service
from ...services.reconstruction.render import render_slice_png
from ...services.reconstruction.slices import (
    PLANES,
    extract_plane,
    plane_counts,
    volume_geometry_summary,
)
from ...services.reconstruction.volume import build_volume
from ...services.validation import checks as checks_module
from .. import deps
from .. import presenters

logger = get_logger("api.reconstruction")

router = APIRouter(prefix="/reconstruction", tags=["reconstruction"])


# ---------------------------------------------------------------------------
# Static routes (declared before the dynamic /{volume_id} route)
# ---------------------------------------------------------------------------

@router.get("/window-presets", summary="Clinical window presets for a modality")
def window_presets(modality: str | None = Query(None)) -> list[dict]:
    return preprocessing_service.presets_for_modality(modality)


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------

@router.post("/create", response_model=JobEnvelope, summary="Reconstruct a 3D volume")
def create(
    request: ReconstructionRequest,
    wait: bool = Query(False, description="Run inline and return the result instead of a job id."),
) -> JobEnvelope:
    record = deps.get_dataset(request.dataset_id)
    analysis = record.analysis
    if analysis is None:
        analysis = series_service.analyze_dataset(record.dataset, record.root)
        record.analysis = analysis

    series_uid = request.series_instance_uid or analysis.primary_series_uid
    if not series_uid:
        raise ValidationError("The dataset does not contain any DICOM series to reconstruct.")
    if request.series_instance_uid is None and len(analysis.series) > 1:
        raise ValidationError(
            "This dataset contains more than one series. Choose one with 'series_instance_uid'.",
            detail="Reconstructing a mix of series would produce a geometrically invalid volume.",
        )

    series_analysis = analysis.series_by_uid(series_uid)
    if series_analysis is None:
        raise ValidationError(f"Series '{series_uid}' is not present in this dataset.")

    uniform = tuple(request.uniform_spacing_mm) if request.uniform_spacing_mm else None

    def job(reporter):
        reporter.step("reconstruction", 0.01, f"Reconstructing volume from series {series_uid}")
        recon = build_volume(
            record.dataset,
            record.root,
            series_analysis,
            allow_instance_number_fallback=request.allow_instance_number_fallback,
            uniform_spacing_mm=uniform,  # type: ignore[arg-type]
            isotropic=request.isotropic,
            progress=reporter,
        )
        reporter.step("persistence", 0.9, "Persisting reconstructed volume")
        volume = recon.volume
        store.save_volume_array(record.dataset_id, volume.volume_id, volume.array)
        # Re-open the array as a read-only memory map so the resident set stays
        # small on laptop-class machines; downstream steps read the same bytes.
        mapped = store.load_volume_array(record.dataset_id, volume.volume_id)
        volume = volume.copy_with(array=mapped, source_id=record.dataset_id)

        findings = checks_module.check_ordering(recon.ordering, series_analysis.info)
        findings.extend(checks_module.check_volume(volume))
        summary = checks_module.summarise(findings)

        report = dict(recon.report)
        report["slice_preview"] = _slice_preview(volume, record.dataset_id)

        volume_record = VolumeRecord(
            dataset_id=record.dataset_id,
            volume=volume,
            ordering=recon.ordering,
            report=report,
            findings=[f.as_dict() for f in findings],
            summary=summary,
            series_instance_uid=series_uid,
        )
        deps.register_volume(volume_record)
        reporter.step("done", 1.0, f"Volume reconstructed: {volume.shape}")
        return presenters.reconstruction_response(volume_record)

    return deps.dispatch("reconstruction", job, wait=wait)


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------

@router.get("/{volume_id}", response_model=dict, summary="Retrieve a reconstructed volume")
def get_volume(volume_id: str) -> dict:
    return presenters.reconstruction_response(deps.get_volume(volume_id))


@router.get("/{volume_id}/geometry", summary="Patient-space geometry of a volume")
def get_geometry(volume_id: str) -> dict:
    record = deps.get_volume(volume_id)
    summary = volume_geometry_summary(record.volume)
    return {
        "geometry": presenters.volume_geometry(record.volume, record.dataset_id).model_dump(mode="json"),
        "summary": summary,
    }


@router.get("/{volume_id}/slice", response_model=SliceResponse, summary="Raw pixel data for one plane")
def get_slice(
    volume_id: str,
    plane: str = Query("axial", pattern="^(axial|coronal|sagittal)$"),
    index: int = Query(0, ge=0),
    window_center: float | None = Query(None),
    window_width: float | None = Query(None),
) -> SliceResponse:
    record = deps.get_volume(volume_id)
    volume = record.volume
    view = extract_plane(volume, plane, index, with_array=True)
    array = view.array
    if array is None:  # pragma: no cover - with_array=True guarantees a value
        raise ValidationError("The requested slice could not be extracted.")

    if array.dtype == "int16":
        dtype = "int16"
        payload = array
    else:
        dtype = "float32"
        payload = array.astype("float32", copy=False)

    encoded = base64.b64encode(np.ascontiguousarray(payload).tobytes()).decode("ascii")
    return SliceResponse(
        volume_id=volume_id,
        plane=plane,  # type: ignore[arg-type]
        index=view.index,
        count=view.count,
        rows=view.rows,
        columns=view.columns,
        row_spacing_mm=view.row_spacing_mm,
        column_spacing_mm=view.column_spacing_mm,
        position_mm=view.position_mm,
        position_label=view.position_label,
        labels=view.labels,
        dtype=dtype,  # type: ignore[arg-type]
        data=encoded,
        intensity_min=view.intensity_min,
        intensity_max=view.intensity_max,
        intensity_units=volume.intensity_units,
        window_center=window_center if window_center is not None else volume.window_center,
        window_width=window_width if window_width is not None else volume.window_width,
        warnings=list(view.warnings),
    )


@router.get("/{volume_id}/thumbnail", summary="Server-rendered PNG preview of a slice")
def get_thumbnail(
    volume_id: str,
    plane: str = Query("axial", pattern="^(axial|coronal|sagittal)$"),
    index: int | None = Query(None, ge=0),
    max_size: int = Query(192, ge=32, le=1024),
    window_center: float | None = Query(None),
    window_width: float | None = Query(None),
) -> Response:
    record = deps.get_volume(volume_id)
    volume = record.volume
    counts = plane_counts(volume)
    count = counts.get(plane, 0)
    resolved = count // 2 if index is None else min(index, max(count - 1, 0))
    if count == 0:
        raise ValidationError(f"The volume has no slices along the {plane} plane.")

    view = extract_plane(volume, plane, resolved, with_array=True)
    if view.array is None:  # pragma: no cover
        raise ValidationError("The requested slice could not be extracted.")
    png = render_slice_png(
        view.array,
        center=window_center if window_center is not None else volume.window_center,
        width=window_width if window_width is not None else volume.window_width,
        max_size=max_size,
    )
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "no-store", "X-Slice-Index": str(resolved), "X-Slice-Count": str(count)},
    )


# ---------------------------------------------------------------------------
# Optional preprocessing (returns a new, independent volume)
# ---------------------------------------------------------------------------

@router.post("/preprocess", response_model=JobEnvelope, summary="Preprocess a volume into a new volume")
def preprocess(
    request: PreprocessRequest,
    wait: bool = Query(False),
) -> JobEnvelope:
    record = deps.get_volume(request.volume_id)

    def job(reporter):
        options = preprocessing_service.PreprocessingOptions(
            normalize=request.normalize,
            denoise=request.denoise,
            denoise_sigma=request.denoise_sigma,
            denoise_size=request.denoise_size,
            clip_min=request.clip_min,
            clip_max=request.clip_max,
            resample_target_mm=tuple(request.resample_target_mm) if request.resample_target_mm else None,
            isotropic=request.isotropic,
            downsample_factor=request.downsample_factor,
        )
        processed = preprocessing_service.preprocess_volume(record.volume, options, progress=reporter)

        if not request.replace_original:
            from ...core.storage import new_id

            processed = processed.copy_with(volume_id=new_id("vol"))

        store.save_volume_array(record.dataset_id, processed.volume_id, processed.array)
        mapped = store.load_volume_array(record.dataset_id, processed.volume_id)
        processed = processed.copy_with(array=mapped)

        if request.replace_original:
            record.volume = processed
            volume_record = record
        else:
            volume_record = VolumeRecord(
                dataset_id=record.dataset_id,
                volume=processed,
                report={"note": "Derived from preprocessing; see the source volume report."},
                summary=checks_module.summarise([]),
                series_instance_uid=record.series_instance_uid,
            )
        deps.register_volume(volume_record)
        reporter.step("done", 1.0, f"Preprocessed volume ready: {processed.shape}")
        return presenters.reconstruction_response(volume_record)

    return deps.dispatch("preprocess", job, wait=wait)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _slice_preview(volume, dataset_id: str) -> list[dict]:
    counts = plane_counts(volume)
    preview: list[dict] = []
    for plane in PLANES:
        count = counts.get(plane, 0)
        if count <= 0:
            continue
        index = count // 2
        view = extract_plane(volume, plane, index, with_array=True)
        if view.array is None:  # pragma: no cover
            continue
        png = render_slice_png(
            view.array,
            center=volume.window_center,
            width=volume.window_width,
            max_size=192,
        )
        preview.append(
            {
                "plane": plane,
                "index": index,
                "count": count,
                "position_label": view.position_label,
                "position_mm": round(float(view.position_mm), 3),
                "png_base64": base64.b64encode(png).decode("ascii"),
            }
        )
    return preview


__all__ = ["router"]
