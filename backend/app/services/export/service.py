"""Export of volumes, masks, meshes and reports.

Every geometry-bearing export carries its spatial metadata:

* NIfTI files are written with a real affine derived from the volume's
  ``origin``/``direction``/``spacing``, so they load correctly oriented in any
  DICOM/NIfTI viewer.
* Mask archives (``.npz``) store the bit-packed mask *and* its shape so they can
  be re-loaded as ground truth.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from ...core.errors import ValidationError
from ...core.logging_ import get_logger
from ...models import MeshAsset, SegmentationResult, Volume

logger = get_logger("export.service")

#: LPS -> RAS flip used for NIfTI output.
_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])


def volume_affine(volume: Volume) -> np.ndarray:
    """Affine mapping voxel indices ``(i, j, k)`` to NIfTI RAS millimetres."""
    direction = np.asarray(volume.direction, dtype=np.float64)
    spacing = np.asarray(volume.spacing, dtype=np.float64)
    origin = np.asarray(volume.origin, dtype=np.float64)
    # columns for (i, j, k) = (column, row, slice) index directions
    matrix_lps = np.column_stack(
        [direction[:, 2] * spacing[2], direction[:, 1] * spacing[1], direction[:, 0] * spacing[0]]
    )
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = _LPS_TO_RAS @ matrix_lps
    affine[:3, 3] = _LPS_TO_RAS @ origin
    return affine


def export_volume_nifti(volume: Volume, path: Path) -> Path:
    try:
        import nibabel as nib
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            "NIfTI export requires the 'nibabel' package.",
            detail=f"pip install nibabel ({exc})",
        ) from exc

    array = np.asarray(volume.array)
    data = np.transpose(array, (2, 1, 0)).astype(np.float32)
    image = nib.Nifti1Image(data, volume_affine(volume))
    image.header.set_xyzt_units("mm")
    image.header["descrip"] = b"ANATO3D reconstructed volume"
    nib.save(image, str(path))
    return path


def export_volume_npz(volume: Volume, path: Path) -> Path:
    np.savez_compressed(
        path,
        array=np.asarray(volume.array),
        spacing=np.asarray(volume.spacing, dtype=np.float64),
        origin=np.asarray(volume.origin, dtype=np.float64),
        direction=np.asarray(volume.direction, dtype=np.float64),
        modality=volume.modality,
        orientation_label=volume.orientation_label,
        intensity_units=volume.intensity_units,
    )
    return path


def export_mask_npz(mask: np.ndarray, path: Path) -> Path:
    mask = np.asarray(mask, dtype=bool)
    np.savez_compressed(path, mask=np.packbits(mask.reshape(-1)), shape=np.asarray(mask.shape, dtype=np.int64))
    return path


def export_mask_nifti(mask: np.ndarray, volume: Volume, path: Path) -> Path:
    try:
        import nibabel as nib
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            "NIfTI export requires the 'nibabel' package.",
            detail=f"pip install nibabel ({exc})",
        ) from exc

    data = np.transpose(np.asarray(mask, dtype=np.uint8), (2, 1, 0))
    image = nib.Nifti1Image(data, volume_affine(volume))
    image.header.set_xyzt_units("mm")
    image.header["descrip"] = b"ANATO3D segmentation mask"
    nib.save(image, str(path))
    return path


def export_json(payload: dict[str, Any], path: Path) -> Path:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def export_mesh_asset(mesh: MeshAsset, path: Path, fmt: str) -> Path:
    from ..mesh.exporters import export_mesh

    if not mesh.mesh or mesh.mesh.n_faces == 0:
        raise ValidationError("The mesh has no geometry to export.")
    return export_mesh(mesh.mesh, path, fmt, color=mesh.color, name=mesh.label or "structure")


def build_bundle(
    target: Path,
    *,
    volume: Volume | None,
    segmentation: SegmentationResult | None,
    mesh: MeshAsset | None,
    reports: dict[str, dict[str, Any]],
    include_volume: bool = True,
    include_mesh: bool = True,
) -> Path:
    """Write a ZIP containing every requested artefact plus a manifest."""
    manifest: dict[str, Any] = {"application": "ANATO3D", "contents": []}
    scratch = io.BytesIO()

    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if volume is not None and include_volume:
            try:
                data = np.transpose(np.asarray(volume.array), (2, 1, 0)).astype(np.float32)
                tmp = io.BytesIO()
                np.save(tmp, data)
                archive.writestr("volume/reconstructed_volume.npy", tmp.getvalue())
                manifest["contents"].append("volume/reconstructed_volume.npy")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Volume export in bundle failed: %s", exc)

            archive.writestr(
                "volume/reconstruction_metadata.json",
                json.dumps(
                    {
                        "dimensions": list(volume.shape),
                        "spacing_mm": [float(v) for v in volume.spacing],
                        "origin_lps_mm": [float(v) for v in volume.origin],
                        "direction": np.asarray(volume.direction).tolist(),
                        "orientation": volume.orientation_label,
                        "patient_position": volume.patient_position,
                        "modality": volume.modality,
                        "intensity_units": volume.intensity_units,
                        "resampled": bool(volume.is_resampled),
                    },
                    indent=2,
                ),
            )
            manifest["contents"].append("volume/reconstruction_metadata.json")

        if segmentation is not None:
            scratch = io.BytesIO()
            np.savez_compressed(
                scratch, mask=np.packbits(np.asarray(segmentation.mask, dtype=bool).reshape(-1)),
                shape=np.asarray(segmentation.mask.shape, dtype=np.int64),
            )
            archive.writestr("segmentation/mask.npz", scratch.getvalue())
            archive.writestr(
                "segmentation/segmentation_metadata.json",
                json.dumps(
                    {
                        "segmentation_id": segmentation.segmentation_id,
                        "label": segmentation.label,
                        "engine": segmentation.engine,
                        "params": {k: v for k, v in segmentation.params.items() if k != "resolved_seeds"},
                        "notes": segmentation.notes,
                        "warnings": segmentation.warnings,
                    },
                    indent=2,
                    default=_json_default,
                ),
            )
            manifest["contents"].extend(
                ["segmentation/mask.npz", "segmentation/segmentation_metadata.json"]
            )

        if mesh is not None and include_mesh:
            for fmt, source in mesh.files.items():
                try:
                    archive.writestr(f"mesh/structure.{fmt}", Path(source).read_bytes())
                    manifest["contents"].append(f"mesh/structure.{fmt}")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Mesh %s could not be added to the bundle: %s", fmt, exc)

        for name, payload in reports.items():
            archive.writestr(f"reports/{name}.json", json.dumps(payload, indent=2, default=_json_default))
            manifest["contents"].append(f"reports/{name}.json")

        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        archive.writestr(
            "README.txt",
            "ANATO3D export bundle\n"
            "=====================\n\n"
            "This bundle was produced by the ANATO3D research prototype.\n"
            "It is NOT a certified medical device and must not be used for diagnosis.\n\n"
            "Contents are listed in manifest.json.\n"
            "Volume arrays are stored in (i, j, k) = (column, row, slice) order, matching NIfTI.\n"
            "Segmentation masks are bit-packed in mask.npz together with their shape.\n",
        )
    return target
