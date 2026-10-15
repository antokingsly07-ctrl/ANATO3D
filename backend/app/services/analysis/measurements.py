"""Physical measurements derived from a mask and its surface.

Two volumes are deliberately reported side by side:

``voxel_volume_mm3``
    ``mask voxel count x voxel volume``.  This is the authoritative measurement:
    it depends only on the segmentation and on the (verified) voxel spacing.

``mesh_volume_mm3``
    The enclosed volume of the triangulated surface, computed with the divergence
    theorem.  It only exists when the surface is watertight; comparing the two is
    a useful reconstruction quality signal.

All lengths are millimetres, areas mm^2 and volumes mm^3 in the API payloads;
the UI converts to cm^3 / cm^2 for display only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

from ...core.logging_ import get_logger
from ...models import MeshData, Volume, view_to_lps

logger = get_logger("analysis.measurements")


def _trimesh(mesh: MeshData) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )


def _pca_extents(points: np.ndarray) -> list[float]:
    if points.shape[0] < 4:
        return []
    centred = points - points.mean(axis=0)
    covariance = np.cov(centred, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    extents = []
    for axis in order:
        direction = eigenvectors[:, axis]
        projection = centred @ direction
        extents.append(float(projection.max() - projection.min()))
    return extents


def compute_measurements(
    mesh: MeshData,
    mask: np.ndarray,
    volume: Volume,
    *,
    refinement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute every measurement the viewer and the analysis panel display."""
    tm = _trimesh(mesh)
    voxel_volume = float(volume.voxel_volume_mm3)
    voxel_count = int(np.count_nonzero(mask))
    voxel_volume_mm3 = voxel_count * voxel_volume

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    view_min = vertices.min(axis=0)
    view_max = vertices.max(axis=0)
    view_centroid = vertices.mean(axis=0)

    lps_vertices = view_to_lps(vertices)
    lps_min = lps_vertices.min(axis=0)
    lps_max = lps_vertices.max(axis=0)

    try:
        area_mm2 = float(tm.area)
    except Exception:  # noqa: BLE001
        area_mm2 = float("nan")

    mesh_volume_mm3: float | None
    try:
        mesh_volume_mm3 = abs(float(tm.volume)) if tm.is_watertight else None
    except Exception:  # noqa: BLE001
        mesh_volume_mm3 = None

    # in-plane resolution used for cross-section areas
    plane_counts = {
        "axial_mm2": float(volume.spacing[0] * volume.spacing[1]),
        "coronal_mm2": float(volume.spacing[0] * volume.spacing[2]),
        "sagittal_mm2": float(volume.spacing[1] * volume.spacing[2]),
    }
    cross_sections: dict[str, Any] = {}
    axis_area = {"axial": plane_counts["axial_mm2"], "coronal": plane_counts["coronal_mm2"], "sagittal": plane_counts["sagittal_mm2"]}
    for axis, name in enumerate(("axial", "coronal", "sagittal")):
        profile = np.count_nonzero(mask, axis=axis).astype(np.float64) * axis_area[name]
        if profile.size:
            cross_sections[name] = {
                "max_area_mm2": round(float(profile.max()), 3),
                "mean_area_mm2": round(float(profile.mean()), 3),
                "min_nonzero_area_mm2": round(float(profile[profile > 0].min()), 3) if np.any(profile > 0) else 0.0,
                "slices_with_foreground": int(np.count_nonzero(profile)),
            }

    sphericity: float | None = None
    if voxel_volume_mm3 > 0 and area_mm2 and np.isfinite(area_mm2) and area_mm2 > 0:
        sphericity = float((np.pi ** (1.0 / 3.0)) * ((6.0 * voxel_volume_mm3) ** (2.0 / 3.0)) / area_mm2)

    dimensions_mm = (view_max - view_min).tolist()

    payload: dict[str, Any] = {
        "mesh": {
            "vertex_count": int(mesh.n_vertices),
            "face_count": int(mesh.n_faces),
            "surface_area_mm2": round(area_mm2, 4) if np.isfinite(area_mm2) else None,
            "mesh_volume_mm3": round(mesh_volume_mm3, 4) if mesh_volume_mm3 is not None else None,
            "is_watertight": bool(tm.is_watertight),
            "is_winding_consistent": bool(tm.is_winding_consistent),
            "euler_number": int(tm.euler_number),
            "body_count": int(tm.body_count),
        },
        "voxel": {
            "count": voxel_count,
            "voxel_volume_mm3": round(voxel_volume, 8),
            "volume_mm3": round(voxel_volume_mm3, 4),
        },
        "dimensions": {
            "view_min": [round(float(v), 4) for v in view_min],
            "view_max": [round(float(v), 4) for v in view_max],
            "size_mm": [round(float(v), 4) for v in dimensions_mm],
            "size_mm_order": "x (patient left/right) x y (patient superior/inferior) x z (patient anterior/posterior)",
            "lps_min": [round(float(v), 4) for v in lps_min],
            "lps_max": [round(float(v), 4) for v in lps_max],
        },
        "centroid": {
            "view_mm": [round(float(v), 4) for v in view_centroid],
            "lps_mm": [round(float(v), 4) for v in view_to_lps(view_centroid)],
        },
        "shape_descriptors": {
            "sphericity": round(sphericity, 5) if sphericity is not None else None,
            "principal_extents_mm": [round(v, 4) for v in _pca_extents(vertices)],
        },
        "cross_sections": cross_sections,
    }

    if refinement is not None:
        payload["refinement"] = refinement

    return payload


def intensity_statistics_in_mask(volume: Volume, mask: np.ndarray) -> dict[str, float]:
    values = np.asarray(volume.array)[mask]
    if values.size == 0:
        return {}
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p5": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "unit": volume.intensity_units,
    }


def measurement_summary(measurements: dict[str, Any]) -> list[dict[str, Any]]:
    """Human-readable measurement list for the analysis panel and reports."""
    mesh = measurements.get("mesh", {})
    voxel = measurements.get("voxel", {})
    dims = measurements.get("dimensions", {})
    centroid = measurements.get("centroid", {})
    shape = measurements.get("shape_descriptors", {})

    volume_mm3 = voxel.get("volume_mm3") or 0.0
    area_mm2 = mesh.get("surface_area_mm2")
    size = dims.get("size_mm") or [0.0, 0.0, 0.0]

    items = [
        {
            "key": "volume",
            "label": "Segmented volume",
            "value": round(volume_mm3 / 1000.0, 4),
            "unit": "cm3",
            "detail": f"{volume_mm3:.1f} mm3 = {voxel.get('count', 0)} voxels "
                      f"x {voxel.get('voxel_volume_mm3', 0):.5f} mm3",
        },
        {
            "key": "surface_area",
            "label": "Surface area",
            "value": round(area_mm2 / 100.0, 4) if area_mm2 else None,
            "unit": "cm2",
            "detail": f"{area_mm2:.1f} mm2 on {mesh.get('face_count', 0)} triangles" if area_mm2 else "unavailable",
        },
        {
            "key": "dimensions",
            "label": "Bounding-box dimensions (L-R x S-I x A-P)",
            "value": [round(v, 2) for v in size],
            "unit": "mm",
            "detail": f"{size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} mm",
        },
        {
            "key": "mesh_volume",
            "label": "Enclosed mesh volume",
            "value": round((mesh.get("mesh_volume_mm3") or 0.0) / 1000.0, 4) if mesh.get("mesh_volume_mm3") else None,
            "unit": "cm3",
            "detail": (
                "Watertight surface; should be close to the voxel volume."
                if mesh.get("is_watertight")
                else "Not available: the surface is open (touches the volume border)."
            ),
        },
        {
            "key": "centroid",
            "label": "Centroid (patient LPS)",
            "value": centroid.get("lps_mm"),
            "unit": "mm",
            "detail": "L = Left(+X), P = Posterior(+Y), S = Superior(+Z)",
        },
        {
            "key": "sphericity",
            "label": "Sphericity",
            "value": shape.get("sphericity"),
            "unit": "1 (perfect sphere)",
            "detail": "1.0 for a sphere; lower values indicate a more irregular or elongated shape.",
        },
    ]
    return items
