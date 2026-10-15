"""Extraction of anatomically labelled axial / coronal / sagittal planes.

The three canonical planes are defined by their *patient-space* normal and by a
screen convention (which patient direction is up / right).  The array axis whose
direction cosine best matches the plane normal becomes the slice axis; the other
two axes are mapped to the screen rows and columns with the appropriate flips.

Screen conventions used (standard radiological viewing):

======  ==============  ================  ============  ===============
plane   into-screen     screen right      screen up     viewer located
======  ==============  ================  ============  ===============
axial   +Z (Superior)   +X (Left)         -Y (Anterior) inferior
coronal +Y (Posterior)  +X (Left)         +Z (Superior) anterior
sagittal -X (Right)     +Y (Posterior)    +Z (Superior) patient's left
======  ==============  ================  ============  ===============
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...core.errors import ValidationError
from ...models import Volume

PLANES = ("axial", "coronal", "sagittal")

#: plane -> (normal, screen-right direction, row-index direction) in LPS.
#:
#: The third vector is the direction in which the *image row index* increases,
#: i.e. downwards on screen.  The top edge of the image therefore corresponds to
#: the opposite patient direction.
PLANE_DEFINITIONS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = {
    "axial": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "coronal": ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    "sagittal": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
}

_AXIS_LETTERS = {0: ("L", "R"), 1: ("P", "A"), 2: ("S", "I")}


def direction_letter(vector: np.ndarray) -> str:
    """Short patient-direction label (``A``/``P``/``L``/``R``/``S``/``I``)."""
    vec = np.asarray(vector, dtype=np.float64)
    idx = int(np.argmax(np.abs(vec)))
    positive, negative = _AXIS_LETTERS[idx]
    return positive if vec[idx] >= 0 else negative


def direction_name(vector: np.ndarray) -> str:
    vec = np.asarray(vector, dtype=np.float64)
    idx = int(np.argmax(np.abs(vec)))
    positive, negative = _AXIS_LETTERS[idx]
    full = {
        0: ("Left", "Right"),
        1: ("Posterior", "Anterior"),
        2: ("Superior", "Inferior"),
    }[idx]
    return full[0] if vec[idx] >= 0 else full[1]


@dataclass
class PlaneAxes:
    plane: str
    slice_axis: int
    row_axis: int
    column_axis: int
    row_sign: float
    column_sign: float
    obliquity_deg: float


@dataclass
class PlaneView:
    plane: str
    index: int
    count: int
    rows: int
    columns: int
    row_spacing_mm: float
    column_spacing_mm: float
    position_mm: float
    position_label: str
    intensity_min: float
    intensity_max: float
    labels: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    array: np.ndarray | None = None


def plane_axes(volume: Volume, plane: str) -> PlaneAxes:
    if plane not in PLANE_DEFINITIONS:
        raise ValidationError(f"Unknown plane '{plane}'. Expected one of {', '.join(PLANES)}.")
    normal, right, up = (np.asarray(v, dtype=np.float64) for v in PLANE_DEFINITIONS[plane])
    direction = np.asarray(volume.direction, dtype=np.float64)
    axes = [direction[:, a] for a in range(3)]

    alignment = [abs(float(np.dot(axes[a], normal))) for a in range(3)]
    slice_axis = int(np.argmax(alignment))
    remaining = [a for a in range(3) if a != slice_axis]

    right_scores = [(a, float(np.dot(axes[a], right))) for a in remaining]
    column_axis, column_sign = max(right_scores, key=lambda item: abs(item[1]))
    row_axis = next(a for a in remaining if a != column_axis)
    row_sign = float(np.dot(axes[row_axis], up))

    return PlaneAxes(
        plane=plane,
        slice_axis=slice_axis,
        row_axis=row_axis,
        column_axis=column_axis,
        row_sign=1.0 if row_sign >= 0 else -1.0,
        column_sign=1.0 if column_sign >= 0 else -1.0,
        obliquity_deg=float(np.degrees(np.arccos(np.clip(alignment[slice_axis], 0.0, 1.0)))),
    )


def plane_counts(volume: Volume) -> dict[str, int]:
    counts: dict[str, int] = {}
    for plane in PLANES:
        axes = plane_axes(volume, plane)
        counts[plane] = int(volume.shape[axes.slice_axis])
    return counts


def extract_plane(volume: Volume, plane: str, index: int, *, with_array: bool = True) -> PlaneView:
    axes = plane_axes(volume, plane)
    count = int(volume.shape[axes.slice_axis])
    if count == 0:
        raise ValidationError(f"The reconstructed volume has no extent along the {plane} axis.")
    index = int(np.clip(index, 0, count - 1))

    data = np.asarray(volume.array)
    sl = np.take(data, index, axis=axes.slice_axis)
    remaining = [a for a in range(3) if a != axes.slice_axis]
    permutation = [remaining.index(axes.row_axis), remaining.index(axes.column_axis)]
    image = np.transpose(sl, permutation)
    if axes.row_sign < 0:
        image = image[::-1, :]
    if axes.column_sign < 0:
        image = image[:, ::-1]

    spacing = volume.spacing
    row_spacing = float(spacing[axes.row_axis])
    column_spacing = float(spacing[axes.column_axis])

    # position of this plane along its normal, in LPS millimetres
    origin = np.asarray(volume.origin, dtype=np.float64)
    slice_dir = np.asarray(volume.direction, dtype=np.float64)[:, axes.slice_axis]
    point = origin + index * spacing[axes.slice_axis] * slice_dir
    normal = np.asarray(PLANE_DEFINITIONS[plane][0], dtype=np.float64)
    position = float(np.dot(point, normal))
    axis_letter = "XYZ"[int(np.argmax(np.abs(normal)))]

    _, right, up = (np.asarray(v, dtype=np.float64) for v in PLANE_DEFINITIONS[plane])
    labels = {
        "top": direction_letter(-up),
        "bottom": direction_letter(up),
        "left": direction_letter(-right),
        "right": direction_letter(right),
        "top_name": direction_name(-up),
        "bottom_name": direction_name(up),
        "left_name": direction_name(-right),
        "right_name": direction_name(right),
    }

    warnings: list[str] = []
    if axes.obliquity_deg > 5.0:
        warnings.append(
            f"The {plane} plane is oblique by {axes.obliquity_deg:.1f}° relative to the patient axis; "
            "the displayed plane is a nearest-axis approximation of the acquisition geometry."
        )

    values = np.asarray(image, dtype=np.float32)
    view = PlaneView(
        plane=plane,
        index=index,
        count=count,
        rows=int(image.shape[0]),
        columns=int(image.shape[1]),
        row_spacing_mm=row_spacing,
        column_spacing_mm=column_spacing,
        position_mm=position,
        position_label=f"{axis_letter} = {position:.2f} mm (LPS)",
        intensity_min=float(np.min(values)),
        intensity_max=float(np.max(values)),
        labels=labels,
        warnings=warnings,
        array=image if with_array else None,
    )
    return view


def extract_mask_plane(mask: np.ndarray, volume: Volume, plane: str, index: int) -> np.ndarray:
    """Extract the same plane from a boolean mask, matching :func:`extract_plane`."""
    axes = plane_axes(volume, plane)
    count = int(volume.shape[axes.slice_axis])
    index = int(np.clip(index, 0, max(count - 1, 0)))
    sl = np.take(np.asarray(mask), index, axis=axes.slice_axis)
    remaining = [a for a in range(3) if a != axes.slice_axis]
    permutation = [remaining.index(axes.row_axis), remaining.index(axes.column_axis)]
    image = np.transpose(sl, permutation)
    if axes.row_sign < 0:
        image = image[::-1, :]
    if axes.column_sign < 0:
        image = image[:, ::-1]
    return np.ascontiguousarray(image.astype(bool))


def volume_geometry_summary(volume: Volume) -> dict[str, object]:
    counts = plane_counts(volume)
    return {
        "dimensions": list(volume.shape),
        "spacing_mm": [round(float(v), 6) for v in volume.spacing],
        "orientation": volume.orientation_label,
        "patient_position": volume.patient_position,
        "plane_slice_counts": counts,
        "origin_lps": [round(float(v), 4) for v in volume.origin],
        "direction": np.asarray(volume.direction).round(6).tolist(),
        "intensity_units": volume.intensity_units,
        "window_center": volume.window_center,
        "window_width": volume.window_width,
        "is_resampled": bool(volume.is_resampled),
        "is_normalized": bool(volume.is_normalized),
        "voxel_volume_mm3": round(volume.voxel_volume_mm3, 6),
        "physical_size_mm": [round(float(v), 3) for v in volume.physical_size_mm],
        "warnings": list(volume.warnings),
        "notes": list(volume.notes),
    }
