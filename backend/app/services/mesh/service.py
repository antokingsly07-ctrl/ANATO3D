"""Mesh generation orchestration: mask -> surface -> refinement -> files."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ...core.errors import MeshGenerationError
from ...core.logging_ import get_logger
from ...core.storage import new_id
from ...models import MeshAsset, SegmentationResult, Volume
from ..analysis.measurements import compute_measurements, intensity_statistics_in_mask, measurement_summary
from .exporters import export_mesh, hex_to_rgba
from .extract import extract_mesh
from .refine import MeshRefinementOptions, refine_mesh

logger = get_logger("mesh.service")

#: Formats always written after a successful mesh generation.
ALWAYS_EXPORT = ("glb", "stl", "obj")


def generate_mesh(
    segmentation: SegmentationResult,
    volume: Volume,
    *,
    output_dir: Path,
    options: MeshRefinementOptions | None = None,
    export_formats: tuple[str, ...] = ALWAYS_EXPORT,
    progress=None,
) -> MeshAsset:
    started = time.time()
    options = options or MeshRefinementOptions()
    notes: list[str] = []
    warnings: list[str] = []

    if progress:
        progress.step("mesh_generation", 0.55, "Running Marching Cubes on the refined mask")

    raw = extract_mesh(segmentation.mask, volume)
    notes.append(
        f"Marching Cubes produced {raw.n_vertices} vertices and {raw.n_faces} triangles at "
        f"the true voxel spacing ({volume.spacing[0]:.4g} x {volume.spacing[1]:.4g} x "
        f"{volume.spacing[2]:.4g} mm)."
    )

    if progress:
        progress.step(
            "mesh_refinement",
            0.60,
            f"Refining surface ({raw.n_faces} triangles)",
        )

    refined, report = refine_mesh(raw, options, progress=progress)
    notes.extend(report.steps)
    warnings.extend(report.warnings)

    if report.decimated and options.max_faces and report.output_faces <= options.max_faces:
        notes.append(f"Surface decimated to the requested budget of {options.max_faces} triangles.")

    if progress:
        progress.step(
            "mesh_refinement",
            0.68,
            f"Refined surface: {refined.n_faces} triangles, "
            f"{'watertight' if True else ''}".strip(),
        )

    measurements = compute_measurements(refined, segmentation.mask, volume, refinement=report.as_dict())
    measurements["intensity"] = intensity_statistics_in_mask(volume, segmentation.mask)

    mesh_id = new_id("mesh")
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for fmt in export_formats:
        path = output_dir / f"{mesh_id}.{fmt}"
        try:
            export_mesh(refined, path, fmt, color=segmentation.color, name=segmentation.label or "structure")
            files[fmt] = str(path)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Export to {fmt.upper()} failed: {exc}")
            logger.warning("Export to %s failed: %s", fmt, exc)

    if "glb" not in files:
        raise MeshGenerationError(
            "The surface could not be exported to GLB, which the 3D viewer requires.",
            detail="; ".join(warnings[-2:]) if warnings else None,
        )

    stats: dict[str, Any] = {
        "mesh_id": mesh_id,
        "label": segmentation.label,
        "color": segmentation.color,
        "color_rgba": hex_to_rgba(segmentation.color),
        "engine": segmentation.engine,
        "vertex_count": refined.n_vertices,
        "face_count": refined.n_faces,
        "measurements": measurements,
        "measurement_summary": measurement_summary(measurements),
        "refinement": report.as_dict(),
        "extraction": {
            "raw_vertices": raw.n_vertices,
            "raw_faces": raw.n_faces,
            "isosurface_level": 0.5,
            "voxel_spacing_mm": [float(v) for v in volume.spacing],
            "voxel_volume_mm3": float(volume.voxel_volume_mm3),
        },
        "available_formats": sorted(files.keys()),
        "duration_s": round(time.time() - started, 3),
    }

    if not measurements["mesh"]["is_watertight"]:
        warnings.append(
            "The surface is not watertight (it is open where the structure touches the volume "
            "boundary). Enclosed mesh volume is therefore not reported; the voxel-based volume "
            "remains valid."
        )

    if progress:
        progress.step(
            "mesh_generation",
            0.70,
            f"Mesh ready: {refined.n_faces} triangles, "
            f"surface area {measurements['mesh']['surface_area_mm2']} mm2",
        )

    return MeshAsset(
        mesh_id=mesh_id,
        segmentation_id=segmentation.segmentation_id,
        volume_id=volume.volume_id,
        label=segmentation.label,
        color=segmentation.color,
        mesh=refined,
        stats=stats,
        created_at=time.time(),
        notes=notes,
        warnings=warnings,
        files=files,
    )


def mesh_summary(asset: MeshAsset) -> dict[str, Any]:
    stats = dict(asset.stats)
    stats["notes"] = list(asset.notes)
    stats["warnings"] = list(asset.warnings)
    return stats


def mask_volume_cm3(mask: np.ndarray, voxel_volume_mm3: float) -> float:
    return float(np.count_nonzero(mask) * voxel_volume_mm3 / 1000.0)
