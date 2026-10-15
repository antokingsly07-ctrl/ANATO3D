"""Marching Cubes surface extraction with correct physical scaling.

Coordinate chain
----------------
1. ``skimage.measure.marching_cubes`` produces vertices in *index* space, already
   scaled by the voxel spacing (it is given ``spacing`` directly).
2. Padding of one voxel is removed so the origin is once again voxel ``(0, 0, 0)``.
3. Index -> patient space:  ``p_lps = origin + (index * spacing) @ direction.T``.
4. Patient (LPS) -> viewer space (Y up, right handed) via
   :func:`app.models.domain.lps_to_view`.

Step 3 uses the same convention as :meth:`app.models.domain.Volume.index_to_lps`,
so mesh coordinates and volume coordinates are consistent by construction.

Winding
-------
A standard DICOM direction matrix has determinant -1, while the LPS->viewer map
has determinant +1.  The composite map is therefore orientation-reversing: the
triangle winding produced by Marching Cubes would point inwards.  That is handled
*explicitly* here by swapping the last two vertex indices of every face rather
than by a heuristic normal-repair pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import measure

from ...core.errors import EmptyMaskError, MeshGenerationError
from ...core.logging_ import get_logger
from ...models import VIEW_FROM_LPS, MeshData, Volume, lps_to_view

logger = get_logger("mesh.extract")


@dataclass
class ExtractionOptions:
    #: Marching Cubes iso-level relative to a 0/1 mask.
    level: float = 0.5
    #: Optional step size for a coarser (faster) surface.
    step_size: int = 1
    #: Pad the mask so surfaces touching the volume border are closed.
    pad: bool = True
    #: Allow the resulting surface to be open at the volume border.
    allow_open_surface: bool = True


def extract_mesh(mask: np.ndarray, volume: Volume, options: ExtractionOptions | None = None) -> MeshData:
    options = options or ExtractionOptions()
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != tuple(volume.shape):
        raise MeshGenerationError(
            "Segmentation mask dimensions do not match the reconstructed volume.",
            detail=f"mask {mask.shape} vs volume {tuple(volume.shape)}",
        )
    if not np.any(mask):
        raise EmptyMaskError(
            "Mesh generation failed because no foreground voxels were detected.",
            detail="Run segmentation again with a wider intensity range.",
        )

    spacing = np.asarray(volume.spacing, dtype=np.float64)

    working = mask
    offset = np.zeros(3, dtype=np.float64)
    if options.pad:
        working = np.pad(mask, 1, mode="constant", constant_values=False)
        offset = spacing.copy()

    try:
        vertices, faces, _normals, _values = measure.marching_cubes(
            working.astype(np.float32),
            level=float(options.level),
            spacing=(float(spacing[0]), float(spacing[1]), float(spacing[2])),
            step_size=int(options.step_size),
            allow_degenerate=False,
        )
    except (ValueError, RuntimeError) as exc:
        raise MeshGenerationError(
            "Marching Cubes could not produce a surface from this mask.",
            detail=str(exc),
        ) from exc

    if vertices.size == 0 or faces.size == 0:
        raise MeshGenerationError(
            "Mesh generation produced no triangles.",
            detail="The mask may be too thin (a single voxel plane) to form a closed surface.",
        )

    index_space = vertices - offset  # (N, 3) in millimetres along (k, j, i)

    direction = np.asarray(volume.direction, dtype=np.float64)
    origin = np.asarray(volume.origin, dtype=np.float64)
    lps = origin + index_space @ direction.T
    view = lps_to_view(lps)

    # Explicit winding correction: the composite transform (patient direction
    # matrix composed with the LPS->viewer rotation) reverses orientation when its
    # determinant is negative, which would invert every triangle normal.
    composite_determinant = float(np.linalg.det(direction)) * float(np.linalg.det(VIEW_FROM_LPS))
    if composite_determinant < 0:
        faces = faces[:, [0, 2, 1]]

    return MeshData(
        vertices=np.ascontiguousarray(view.astype(np.float32)),
        faces=np.ascontiguousarray(faces.astype(np.int32)),
        normals=None,
    )


def lps_vertices_of(mesh: MeshData) -> np.ndarray:
    """Inverse of the viewer transform, for reporting in patient coordinates."""
    from ...models import view_to_lps

    return view_to_lps(mesh.vertices)


def mesh_bounding_box(mesh: MeshData) -> tuple[np.ndarray, np.ndarray]:
    return mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
