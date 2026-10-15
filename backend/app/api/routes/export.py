"""Export endpoints: volume, mask, mesh, reports and a ZIP bundle."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from ...core.errors import NotFoundError, ValidationError
from ...core.logging_ import get_logger
from ...core.storage import new_id, store
from ...schemas.processing import ExportItem, ExportManifest
from ...services import export as export_service
from .. import deps
from .. import presenters

logger = get_logger("api.export")

router = APIRouter(prefix="/export", tags=["export"])

PRIVACY_NOTE = (
    "Exports contain no patient-identifying information: DICOM header tags are never written. "
    "This application is a research prototype and is NOT a certified medical device."
)

_MEDIA = {
    "volume_nifti": ("application/gzip", ".nii.gz"),
    "volume_npz": ("application/octet-stream", ".npz"),
    "mask_nifti": ("application/gzip", "_mask.nii.gz"),
    "mask_npz": ("application/octet-stream", "_mask.npz"),
    "mesh_glb": ("model/gltf-binary", ".glb"),
    "mesh_stl": ("model/stl", ".stl"),
    "mesh_obj": ("text/plain", ".obj"),
    "mesh_ply": ("application/octet-stream", ".ply"),
    "metadata_json": ("application/json", "_metadata.json"),
    "validation_json": ("application/json", "_validation.json"),
    "analysis_json": ("application/json", "_analysis.json"),
    "bundle": ("application/zip", "_bundle.zip"),
}


@router.get("/{dataset_id}/manifest", response_model=ExportManifest, summary="List available exports")
def manifest(
    dataset_id: str,
    volume_id: str | None = Query(None),
    segmentation_id: str | None = Query(None),
    mesh_id: str | None = Query(None),
    validation_id: str | None = Query(None),
) -> ExportManifest:
    deps.get_dataset(dataset_id)
    query = _query_string(volume_id=volume_id, segmentation_id=segmentation_id, mesh_id=mesh_id, validation_id=validation_id)
    base = f"/api/export/{dataset_id}/file"

    items: list[ExportItem] = []

    def add(key: str, label: str, description: str, available: bool) -> None:
        items.append(
            ExportItem(
                key=key,
                label=label,
                description=description,
                url=f"{base}/{key}{query}" if available else None,
                available=available,
                media_type=_MEDIA.get(key, (None, None))[0],
            )
        )

    has_volume = bool(volume_id)
    has_seg = bool(segmentation_id)
    has_mesh = bool(mesh_id)
    has_validation = bool(validation_id)

    add("volume_nifti", "Volume (NIfTI)", "Reconstructed volume with a real affine; loads in any NIfTI viewer.", has_volume)
    add("volume_npz", "Volume (NumPy)", "Raw volume array plus its geometry metadata.", has_volume)
    add("mask_nifti", "Segmentation mask (NIfTI)", "Binary mask aligned to the volume grid.", has_seg and has_volume)
    add("mask_npz", "Segmentation mask (NumPy)", "Bit-packed mask with its shape.", has_seg)
    add("mesh_glb", "3D model (GLB)", "Binary glTF for three.js and most 3D tools.", has_mesh)
    add("mesh_stl", "3D model (STL)", "Triangle mesh for CAD and 3D printing.", has_mesh)
    add("mesh_obj", "3D model (OBJ)", "Wavefront OBJ triangle mesh.", has_mesh)
    add("mesh_ply", "3D model (PLY)", "Polygon file format mesh.", has_mesh)
    add("metadata_json", "Metadata (JSON)", "Dataset summary and reconstruction report.", True)
    add("validation_json", "Validation report (JSON)", "Consistency findings and optional metrics.", has_validation)
    add("analysis_json", "Analysis report (JSON)", "Measurements and intensity statistics.", has_mesh or has_seg)
    add("bundle", "Export bundle (ZIP)", "Everything above in one archive, plus a manifest.", has_volume or has_seg or has_mesh)

    return ExportManifest(dataset_id=dataset_id, items=items, privacy_note=PRIVACY_NOTE)


@router.get("/{dataset_id}/file/{key}", summary="Download one export artefact")
def download(
    dataset_id: str,
    key: str,
    volume_id: str | None = Query(None),
    segmentation_id: str | None = Query(None),
    mesh_id: str | None = Query(None),
    validation_id: str | None = Query(None),
) -> FileResponse:
    deps.get_dataset(dataset_id)
    if key not in _MEDIA:
        raise ValidationError(
            f"Unknown export key '{key}'.",
            detail=f"Valid keys: {', '.join(sorted(_MEDIA))}.",
        )

    workspace = store.workspace(dataset_id)
    out_dir = workspace.derived_dir / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    volume_record = deps.find_volume(volume_id) if volume_id else None
    seg_record = deps.find_segmentation(segmentation_id) if segmentation_id else None
    mesh_record = deps.find_mesh(mesh_id) if mesh_id else None
    validation_record = deps.find_validation(validation_id) if validation_id else None
    if validation_id and validation_record is None:
        raise NotFoundError("That validation report is not available.")

    stem = "anato3d"
    media_type, suffix = _MEDIA[key]

    if key == "bundle":
        target = out_dir / f"{new_id('bundle')}{suffix}"
        reports: dict = {}
        if validation_record is not None:
            reports["validation"] = validation_record.report
        if mesh_record is not None:
            reports["analysis"] = deps.find_analysis(mesh_record.asset.mesh_id).payload  # type: ignore[union-attr]
        export_service.build_bundle(
            target,
            volume=volume_record.volume if volume_record else None,
            segmentation=seg_record.result if seg_record else None,
            mesh=mesh_record.asset if mesh_record else None,
            reports=reports,
        )
        return FileResponse(target, media_type=media_type, filename=f"{stem}{suffix}")

    if key in ("volume_nifti", "volume_npz"):
        if volume_record is None:
            raise ValidationError("A volume_id is required for this export.")
        target = out_dir / f"{volume_record.volume_id}_{key}{suffix}"
        if key == "volume_nifti":
            export_service.export_volume_nifti(volume_record.volume, target)
        else:
            export_service.export_volume_npz(volume_record.volume, target)
        return FileResponse(target, media_type=media_type, filename=f"{volume_record.volume_id}{suffix}")

    if key in ("mask_nifti", "mask_npz"):
        if seg_record is None:
            raise ValidationError("A segmentation_id is required for this export.")
        target = out_dir / f"{seg_record.segmentation_id}_{key}{suffix}"
        if key == "mask_nifti":
            volume_record = volume_record or deps.get_volume(seg_record.result.volume_id)
            export_service.export_mask_nifti(seg_record.result.mask, volume_record.volume, target)
        else:
            export_service.export_mask_npz(seg_record.result.mask, target)
        return FileResponse(target, media_type=media_type, filename=f"{seg_record.segmentation_id}{suffix}")

    if key.startswith("mesh_"):
        if mesh_record is None:
            raise ValidationError("A mesh_id is required for this export.")
        fmt = key.split("_", 1)[1]
        existing = mesh_record.asset.files.get(fmt)
        if existing and Path(existing).exists():
            return FileResponse(
                Path(existing), media_type=media_type, filename=f"{mesh_record.asset.mesh_id}.{fmt}"
            )
        target = out_dir / f"{mesh_record.asset.mesh_id}.{fmt}"
        export_service.export_mesh_asset(mesh_record.asset, target, fmt)
        return FileResponse(target, media_type=media_type, filename=f"{mesh_record.asset.mesh_id}{suffix}")

    if key == "metadata_json":
        payload = {
            "dataset": presenters.dataset_summary(deps.get_dataset(dataset_id)).model_dump(mode="json"),
            "privacy_note": PRIVACY_NOTE,
        }
        if volume_record is not None:
            payload["reconstruction"] = volume_record.report
            payload["geometry"] = presenters.volume_geometry(volume_record.volume, dataset_id).model_dump(mode="json")
        if seg_record is not None:
            payload["segmentation"] = presenters.segmentation_response(seg_record)
        target = out_dir / f"{new_id('meta')}{suffix}"
        export_service.export_json(payload, target)
        return FileResponse(target, media_type=media_type, filename=f"{stem}{suffix}")

    if key == "validation_json":
        if validation_record is None:
            raise ValidationError("A validation_id is required for this export.")
        target = out_dir / f"{validation_record.validation_id}{suffix}"
        export_service.export_json(validation_record.report, target)
        return FileResponse(target, media_type=media_type, filename=f"{validation_record.validation_id}{suffix}")

    if key == "analysis_json":
        analysis_id = mesh_id or segmentation_id
        if not analysis_id:
            raise ValidationError("A mesh_id or segmentation_id is required for this export.")
        analysis = deps.get_analysis(analysis_id)
        target = out_dir / f"{analysis_id}_analysis{suffix}"
        export_service.export_json(analysis.payload, target)
        return FileResponse(target, media_type=media_type, filename=f"{analysis_id}_analysis{suffix}")

    raise ValidationError(f"Export key '{key}' is not handled.")  # pragma: no cover


def _query_string(**params: str | None) -> str:
    pairs = [f"{key}={value}" for key, value in params.items() if value]
    return ("?" + "&".join(pairs)) if pairs else ""


__all__ = ["router"]
