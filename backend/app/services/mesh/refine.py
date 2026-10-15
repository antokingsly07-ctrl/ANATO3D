"""Mesh refinement.

Goals, in priority order:

1. keep the anatomy (no aggressive global smoothing that erases thin structures),
2. remove spurious fragments and non-manifold junk,
3. optionally reduce triangle count so the surface can be streamed to a browser.

The default configuration deletes components below a physical size threshold and
applies a small number of *Taubin* (non-shrinking) smoothing iterations.  Taubin
smoothing is used deliberately because a plain Laplacian pass shrinks bone
surfaces, which would corrupt the volume measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import trimesh

from ...core.errors import MeshGenerationError
from ...core.logging_ import get_logger
from ...models import MeshData

logger = get_logger("mesh.refine")

try:  # optional fast decimation backend
    import fast_simplification  # noqa: F401

    HAS_FAST_SIMPLIFICATION = True
except Exception:  # noqa: BLE001
    HAS_FAST_SIMPLIFICATION = False


@dataclass
class MeshRefinementOptions:
    #: Remove connected components whose enclosed volume is below this (mm^3).
    min_component_volume_mm3: float = 25.0
    #: Hard cap on the number of components kept (0 = keep all that pass the size test).
    keep_largest_n: int = 0
    #: Non-shrinking Taubin smoothing iterations (0 disables smoothing).
    smooth_iterations: int = 8
    smooth_lambda: float = 0.5
    smooth_nu: float = -0.53
    #: Decimate to at most this many faces (0 = do not decimate).
    max_faces: int = 0
    #: Alternatively reduce to this fraction of the original face count.
    decimate_ratio: float = 1.0
    #: Merge vertices closer than this (mm) and drop degenerate faces.
    merge_tolerance_mm: float = 1e-4
    recompute_normals: bool = True


@dataclass
class MeshRefinementReport:
    input_vertices: int = 0
    input_faces: int = 0
    output_vertices: int = 0
    output_faces: int = 0
    input_components: int = 0
    output_components: int = 0
    removed_components: int = 0
    removed_faces: int = 0
    decimated: bool = False
    smoothed: bool = False
    steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_vertices": self.input_vertices,
            "input_faces": self.input_faces,
            "output_vertices": self.output_vertices,
            "output_faces": self.output_faces,
            "input_components": self.input_components,
            "output_components": self.output_components,
            "removed_components": self.removed_components,
            "removed_faces": self.removed_faces,
            "decimated": self.decimated,
            "smoothed": self.smoothed,
            "steps": list(self.steps),
            "warnings": list(self.warnings),
        }


def _to_trimesh(mesh: MeshData) -> trimesh.Trimesh:
    tm = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )
    return tm


def _from_trimesh(tm: trimesh.Trimesh, normals: np.ndarray | None = None) -> MeshData:
    return MeshData(
        vertices=np.ascontiguousarray(_safe_transform(tm.vertices).astype(np.float32)),
        faces=np.ascontiguousarray(np.asarray(tm.faces, dtype=np.int32)),
        normals=normals,
    )


def _safe_transform(vertices: np.ndarray) -> np.ndarray:
    """Round-trip vertices through float32 to keep CPU/GPU results identical."""
    return np.asarray(vertices, dtype=np.float32).astype(np.float64)


def _vertex_cluster_decimation(tm: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Dependency-free decimation by vertex clustering.

    Coarser than quadric edge collapse but it preserves the physical scale and
    overall shape, which matters more than triangle quality for measurements.
    """
    if target_faces <= 0 or len(tm.faces) <= target_faces:
        return tm

    vertices = np.asarray(tm.vertices, dtype=np.float64)
    faces = np.asarray(tm.faces, dtype=np.int64)
    lower = vertices.min(axis=0)
    extent = np.maximum(vertices.max(axis=0) - lower, 1e-9)

    best_vertices = vertices
    best_faces = faces
    for fraction in np.linspace(0.004, 0.6, 40):
        cell = np.maximum(extent * float(fraction), 1e-6)
        keys = np.floor((vertices - lower) / cell).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        count = int(inverse.max()) + 1

        summed = np.zeros((count, 3), dtype=np.float64)
        np.add.at(summed, inverse, vertices)
        counts = np.bincount(inverse, minlength=count).astype(np.float64)
        counts[counts == 0] = 1.0
        new_vertices = summed / counts[:, None]

        new_faces = inverse[faces]
        keep = (
            (new_faces[:, 0] != new_faces[:, 1])
            & (new_faces[:, 1] != new_faces[:, 2])
            & (new_faces[:, 0] != new_faces[:, 2])
        )
        new_faces = new_faces[keep]
        if new_faces.size:
            new_faces = np.unique(np.sort(new_faces, axis=1), axis=0)

        best_vertices, best_faces = new_vertices, new_faces
        if len(new_faces) <= target_faces:
            break

    if best_faces.size == 0:
        return tm
    return trimesh.Trimesh(vertices=best_vertices, faces=best_faces, process=False)


def _decimate(tm: trimesh.Trimesh, target_faces: int, report: MeshRefinementReport) -> trimesh.Trimesh:
    if target_faces <= 0 or len(tm.faces) <= target_faces:
        return tm
    if HAS_FAST_SIMPLIFICATION:
        try:
            simplified = tm.simplify_quadric_decimation(face_count=int(target_faces))
            if simplified is not None and len(simplified.faces) > 0:
                report.steps.append(
                    f"Quadric edge-collapse decimation: {len(tm.faces)} -> {len(simplified.faces)} faces."
                )
                report.decimated = True
                return simplified
        except Exception as exc:  # noqa: BLE001
            logger.warning("Quadric decimation failed (%s); falling back to vertex clustering.", exc)
    else:
        report.warnings.append(
            "The 'fast-simplification' package is not installed; using dependency-free vertex "
            "clustering decimation, which produces a coarser surface."
        )
    clustered = _vertex_cluster_decimation(tm, target_faces)
    if clustered is not tm:
        report.steps.append(
            f"Vertex-clustering decimation: {len(tm.faces)} -> {len(clustered.faces)} faces."
        )
        report.decimated = True
    return clustered


def _filter_components(
    tm: trimesh.Trimesh,
    options: MeshRefinementOptions,
    report: MeshRefinementReport,
) -> trimesh.Trimesh:
    try:
        parts = tm.split(only_watertight=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Component split failed: %s", exc)
        return tm

    report.input_components = len(parts)
    if len(parts) <= 1 and options.keep_largest_n != 1:
        report.output_components = len(parts)
        return tm

    scored: list[tuple[float, trimesh.Trimesh]] = []
    for part in parts:
        try:
            volume = abs(float(part.volume)) if part.is_watertight else 0.0
            if volume == 0.0:
                # open surface: use an area-based proxy so thin sheets are not deleted
                volume = float(part.area) * float(options.merge_tolerance_mm or 1e-3)
        except Exception:  # noqa: BLE001
            volume = 0.0
        scored.append((volume, part))

    scored.sort(key=lambda item: item[0], reverse=True)
    kept: list[trimesh.Trimesh] = []
    for rank, (volume, part) in enumerate(scored):
        if options.keep_largest_n > 0 and rank >= options.keep_largest_n:
            break
        if options.keep_largest_n == 0 and volume < options.min_component_volume_mm3:
            continue
        kept.append(part)

    if not kept:
        report.warnings.append(
            "All mesh components were below the minimum component volume; keeping the largest one "
            "so that a surface is still produced."
        )
        kept = [scored[0][1]]

    report.removed_components = len(scored) - len(kept)
    if report.removed_components:
        report.steps.append(
            f"Removed {report.removed_components} fragment(s) smaller than "
            f"{options.min_component_volume_mm3:g} mm3."
        )

    combined = trimesh.util.concatenate(kept)
    report.output_components = len(kept)
    return combined


def refine_mesh(
    mesh: MeshData,
    options: MeshRefinementOptions | None = None,
    *,
    progress=None,
) -> tuple[MeshData, MeshRefinementReport]:
    options = options or MeshRefinementOptions()
    report = MeshRefinementReport()
    tm = _to_trimesh(mesh)
    report.input_vertices = int(len(tm.vertices))
    report.input_faces = int(len(tm.faces))

    if len(tm.faces) == 0:
        raise MeshGenerationError("The surface has no triangles to refine.")

    # 1. hygiene -----------------------------------------------------------
    tm.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=6)
    try:
        tm.update_faces(tm.nondegenerate_faces(height=1e-8))
    except TypeError:
        tm.update_faces(tm.nondegenerate_faces())
    tm.update_faces(tm.unique_faces())
    tm.remove_unreferenced_vertices()
    report.steps.append(
        f"Topology cleanup: {report.input_vertices} -> {len(tm.vertices)} vertices after merging "
        "duplicates and removing degenerate or duplicated faces."
    )
    if progress:
        progress.log(report.steps[-1], stage="mesh_refinement")

    # 2. component filtering ----------------------------------------------
    tm = _filter_components(tm, options, report)
    if progress and report.removed_components:
        progress.log(report.steps[-1], stage="mesh_refinement")

    # 3. normal consistency -------------------------------------------------
    if options.recompute_normals:
        try:
            tm.fix_normals(multibody=True)
            report.steps.append("Fixed triangle winding so normals point outward.")
        except Exception as exc:  # noqa: BLE001
            report.warnings.append(f"Normal orientation could not be fully corrected: {exc}")

    # 4. smoothing ----------------------------------------------------------
    if options.smooth_iterations > 0:
        try:
            trimesh.smoothing.filter_taubin(
                tm,
                lamb=float(options.smooth_lambda),
                nu=float(options.smooth_nu),
                iterations=int(options.smooth_iterations),
            )
            report.smoothed = True
            report.steps.append(
                f"Taubin smoothing ({options.smooth_iterations} iterations, lambda={options.smooth_lambda:g}, "
                f"nu={options.smooth_nu:g}). Taubin is non-shrinking, preserving volume."
            )
        except Exception as exc:  # noqa: BLE001
            report.warnings.append(f"Smoothing skipped: {exc}")
    else:
        report.steps.append("Smoothing disabled; the raw Marching Cubes surface is preserved.")

    # 5. decimation ---------------------------------------------------------
    target = int(options.max_faces or 0)
    if target <= 0 and 0.0 < options.decimate_ratio < 1.0:
        target = max(4, int(len(tm.faces) * float(options.decimate_ratio)))
    if target > 0:
        before = len(tm.faces)
        tm = _decimate(tm, target, report)
        report.removed_faces += before - len(tm.faces)

    tm.remove_unreferenced_vertices()
    if len(tm.faces) == 0:
        raise MeshGenerationError("Mesh refinement removed every triangle.")

    report.output_vertices = int(len(tm.vertices))
    report.output_faces = int(len(tm.faces))
    if not report.output_components:
        report.output_components = 1

    refined = _from_trimesh(tm)
    return refined, report


def mesh_quality(tm: trimesh.Trimesh) -> dict[str, Any]:
    """Surface quality descriptors reported alongside the measurements."""
    try:
        edge_lengths = np.asarray(tm.edges_unique_length, dtype=np.float64)
        stats = {
            "edge_length_mean_mm": round(float(np.mean(edge_lengths)), 4),
            "edge_length_p95_mm": round(float(np.percentile(edge_lengths, 95)), 4),
            "edge_length_max_mm": round(float(np.max(edge_lengths)), 4),
        }
    except Exception:  # noqa: BLE001
        stats = {}
    return {
        "is_watertight": bool(tm.is_watertight),
        "is_winding_consistent": bool(tm.is_winding_consistent),
        "euler_number": int(tm.euler_number),
        "body_count": int(tm.body_count),
        **stats,
    }
