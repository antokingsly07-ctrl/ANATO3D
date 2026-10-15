"""Response presenters.

Each function converts a stored record (or raw domain object) into the Pydantic
model defined in ``app/schemas``.  Keeping this separate from the routers means
the same shapes are produced consistently regardless of which endpoint is used.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..models.domain import DicomMetadata, Volume
from ..models.records import DatasetRecord, VolumeRecord
from ..schemas.dicom import DatasetSummary, SeriesSummary, StudySummary
from ..schemas.processing import MeshResponse, SegmentationResponse, ValidationResponse, VolumeGeometry
from ..schemas.common import FindingModel
from ..services.reconstruction.slices import volume_geometry_summary
from .deps import DISCLAIMER


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def dataset_summary(record: DatasetRecord) -> DatasetSummary:
    dataset = record.dataset
    dicom = dataset.dicom_files
    unreadable = sum(1 for f in dataset.files if f.error and not f.is_dicom)
    non_image = sum(1 for f in dataset.files if f.error and f.is_dicom)

    files = [
        {
            "name": f.original_name,
            "stored_name": f.stored_name,
            "relative_path": f.path,
            "size_bytes": f.size_bytes,
            "is_dicom": f.is_dicom,
            "error": f.error,
        }
        for f in dataset.files
    ]

    warnings = list(record.warnings) + list(dataset.warnings)
    if record.is_synthetic:
        warnings = list(record.synthetic_notes) + warnings

    return DatasetSummary(
        dataset_id=record.dataset_id,
        file_count=dataset.file_count,
        dicom_file_count=len(dicom),
        non_image_file_count=non_image,
        unreadable_file_count=unreadable,
        total_bytes=dataset.total_bytes,
        is_synthetic=record.is_synthetic,
        synthetic_notes=list(record.synthetic_notes),
        files=files,
        warnings=warnings,
    )


def study_summaries(analysis) -> list[StudySummary]:
    out: list[StudySummary] = []
    for study in analysis.studies:
        out.append(
            StudySummary(
                study_instance_uid=str(study.get("study_instance_uid") or "unknown"),
                series_count=int(study.get("series_count") or 0),
                slice_count=int(study.get("slice_count") or 0),
                study_description=study.get("study_description"),  # type: ignore[arg-type]
                modalities=[str(m) for m in (study.get("modalities") or [])],
                series_instance_uids=[str(u) for u in (study.get("series_instance_uids") or [])],
            )
        )
    return out


def series_summaries(analysis) -> list[SeriesSummary]:
    out: list[SeriesSummary] = []
    for entry in analysis.series:
        info = entry.info
        out.append(
            SeriesSummary(
                series_instance_uid=info.series_instance_uid,
                study_instance_uid=info.study_instance_uid,
                modality=info.modality,
                series_number=info.series_number,
                series_description=info.series_description,
                body_part_examined=info.body_part_examined,
                rows=info.rows,
                columns=info.columns,
                slice_count=info.slice_count,
                orientation=info.orientation_label,
                slice_spacing_mm=round(info.slice_spacing, 6) if info.slice_spacing else None,
                irregular_spacing=bool(info.irregular_spacing),
                has_complete_geometry=bool(info.has_complete_geometry),
                is_reconstructable=bool(info.has_complete_geometry and info.slice_count > 0),
                issues=list(info.issues),
                warnings=list(entry.ordering_warnings),
            )
        )
    return out


def metadata_report(meta: DicomMetadata) -> dict[str, Any]:
    def as_list(value: Any) -> list[float] | None:
        if value is None:
            return None
        return [float(v) for v in value]

    return {
        "modality": meta.modality,
        "study_instance_uid": meta.study_instance_uid,
        "series_instance_uid": meta.series_instance_uid,
        "series_description": meta.series_description,
        "study_description": meta.study_description,
        "series_number": meta.series_number,
        "body_part_examined": meta.body_part_examined,
        "patient_position": meta.patient_position,
        "manufacturer": meta.manufacturer,
        "rows": meta.rows,
        "columns": meta.columns,
        "pixel_spacing_mm": as_list(meta.pixel_spacing),
        "slice_thickness_mm": meta.slice_thickness,
        "spacing_between_slices_mm": meta.spacing_between_slices,
        "image_position_patient_mm": as_list(meta.image_position_patient),
        "image_orientation_patient": as_list(meta.image_orientation_patient),
        "rescale_slope": meta.rescale_slope,
        "rescale_intercept": meta.rescale_intercept,
        "window_center": meta.window_center,
        "window_width": meta.window_width,
        "missing_spatial_tags": list(meta.missing_spatial_tags),
    }


def findings_from_dicts(findings: list[dict[str, Any]]) -> list[FindingModel]:
    return [FindingModel.model_validate(f) for f in findings]


# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------

def volume_geometry(volume: Volume, dataset_id: str | None) -> VolumeGeometry:
    summary = volume_geometry_summary(volume)
    return VolumeGeometry(
        volume_id=volume.volume_id,
        dataset_id=dataset_id,
        modality=volume.modality,
        series_instance_uid=volume.series_instance_uid,
        series_description=volume.series_description,
        dimensions=[int(v) for v in volume.shape],
        spacing_mm=[float(v) for v in volume.spacing],
        origin_lps_mm=[float(v) for v in volume.origin],
        direction=np.asarray(volume.direction, dtype=np.float64).round(6).tolist(),
        orientation=volume.orientation_label,
        patient_position=volume.patient_position,
        physical_size_mm=[float(v) for v in volume.physical_size_mm],
        voxel_volume_mm3=float(volume.voxel_volume_mm3),
        intensity_units=volume.intensity_units,
        window_center=volume.window_center,
        window_width=volume.window_width,
        is_resampled=bool(volume.is_resampled),
        is_normalized=bool(volume.is_normalized),
        plane_slice_counts={str(k): int(v) for k, v in (summary.get("plane_slice_counts") or {}).items()},
        warnings=list(volume.warnings),
        notes=list(volume.notes),
    )


def reconstruction_response(record: VolumeRecord) -> dict[str, Any]:
    from ..schemas.processing import ReconstructionResponse
    from ..schemas.common import ValidationSummaryModel

    return ReconstructionResponse(
        volume_id=record.volume_id,
        dataset_id=record.dataset_id,
        report=record.report,
        geometry=volume_geometry(record.volume, record.dataset_id),
        findings=findings_from_dicts(record.findings),
        summary=ValidationSummaryModel.model_validate(record.summary),
        slice_preview=record.report.get("slice_preview", []) if isinstance(record.report, dict) else [],
    ).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segmentation_response(record) -> dict[str, Any]:
    result = record.result
    stats = result.params.get("statistics") if isinstance(result.params, dict) else None
    return SegmentationResponse(
        segmentation_id=result.segmentation_id,
        volume_id=result.volume_id,
        label=result.label,
        color=result.color,
        engine=result.engine,
        requested_engine=result.params.get("requested_engine") if isinstance(result.params, dict) else None,
        statistics=stats if isinstance(stats, dict) else {},
        params=_json_safe(result.params),
        notes=list(result.notes),
        warnings=list(result.warnings),
        findings=findings_from_dicts(getattr(record, "findings", []) or []),
        duration_s=round(float(result.duration_s), 3),
    ).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

def mesh_response(asset) -> dict[str, Any]:
    stats = asset.stats
    return MeshResponse(
        mesh_id=asset.mesh_id,
        segmentation_id=asset.segmentation_id,
        volume_id=asset.volume_id,
        label=asset.label,
        color=asset.color,
        vertex_count=int(stats.get("vertex_count", 0)),
        face_count=int(stats.get("face_count", 0)),
        measurements=_json_safe(stats.get("measurements", {})),
        measurement_summary=_json_safe(stats.get("measurement_summary", [])),
        refinement=_json_safe(stats.get("refinement", {})),
        extraction=_json_safe(stats.get("extraction", {})),
        available_formats=sorted(asset.files.keys()),
        notes=list(asset.notes),
        warnings=list(asset.warnings),
        duration_s=round(float(stats.get("duration_s", 0.0)), 3),
    ).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validation_response(report: dict[str, Any]) -> dict[str, Any]:
    return ValidationResponse(
        validation_id=str(report.get("validation_id", "")),
        summary=report.get("summary", {}),
        findings=findings_from_dicts(report.get("findings", [])),
        ground_truth=report.get("ground_truth", {}),
        dataset_information=report.get("dataset_information", {}),
        interpretation_notes=list(report.get("interpretation_notes", [])),
        disclaimer=str(report.get("disclaimer", DISCLAIMER)),
        duration_s=float(report.get("duration_s", 0.0)),
    ).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analysis_payload(
    asset,
    volume: Volume,
    segmentation,
) -> dict[str, Any]:
    measurements = (asset.stats or {}).get("measurements", {})
    stats = None
    if segmentation is not None and isinstance(segmentation.params, dict):
        candidate = segmentation.params.get("statistics")
        if isinstance(candidate, dict):
            stats = candidate
    return {
        "id": asset.mesh_id,
        "kind": "mesh",
        "mesh_id": asset.mesh_id,
        "segmentation_id": asset.segmentation_id,
        "volume_id": asset.volume_id,
        "label": asset.label,
        "color": asset.color,
        "engine": segmentation.engine if segmentation is not None else None,
        "measurements": _json_safe(measurements),
        "measurement_summary": _json_safe((asset.stats or {}).get("measurement_summary", [])),
        "mesh": _json_safe(measurements.get("mesh", {})),
        "voxel": _json_safe(measurements.get("voxel", {})),
        "intensity": _json_safe(measurements.get("intensity", {})),
        "segmentation_statistics": _json_safe(stats or {}),
        "volume_geometry": volume_geometry(volume, volume.source_id).model_dump(mode="json"),
        "notes": list(asset.notes),
        "warnings": list(asset.warnings),
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _json_safe(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays to plain Python objects."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "analysis_payload",
    "dataset_summary",
    "findings_from_dicts",
    "mesh_response",
    "metadata_report",
    "reconstruction_response",
    "segmentation_response",
    "series_summaries",
    "study_summaries",
    "validation_response",
    "volume_geometry",
]
