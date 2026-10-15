"""Spatial ordering of DICOM slices.

This module is the heart of the "never trust filenames" requirement.  Ordering
is derived exclusively from patient-space geometry:

``ImageOrientationPatient`` (IOP) gives two direction cosines defining the
in-plane basis of every slice::

    d_i = IOP[0:3]   direction of increasing *column* index
    d_j = IOP[3:6]   direction of increasing *row* index
    d_k = d_i x d_j  slice normal

``ImagePositionPatient`` (IPP) gives the patient-space position of the centre of
the first voxel of the slice.  Projecting IPP onto the slice normal yields a
scalar position in millimetres along the acquisition axis; sorting by that
scalar gives the physically correct stack order regardless of filenames,
``InstanceNumber`` or how the files were exported.

Direction matrix convention (matching ITK/GDCM)::

    direction = column_stack([d_k, d_j, d_i])

so that ``physical = origin + direction @ (index * spacing)`` with
``index = (k, j, i)``.  Note that this matrix has determinant ``-1`` for a
normal right-handed DICOM series, exactly as in ITK.  Downstream code accounts
for the resulting orientation reversal explicitly when needed (see
``services/mesh/extract.py``) rather than "fixing" it by silently flipping the
data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...core.errors import DicomError, IncompleteOrientationError
from ...core.logging_ import get_logger
from ...models import SliceGeometry

logger = get_logger("dicom.ordering")

#: Cosine of 15 degrees -- a normal within this of a patient axis is considered
#: "axis aligned" for the purpose of labelling.
AXIS_ALIGNED_COS = float(np.cos(np.radians(15.0)))
#: Two orientation vectors closer than this are considered identical.
IOP_TOLERANCE = 1e-3
#: Relative deviation from the median slice spacing above which the series is
#: labelled irregular.
IRREGULARITY_TOLERANCE = 0.05
#: Duplicate slice positions closer than this (mm) are flagged.
DUPLICATE_TOLERANCE_MM = 1e-3
#: Tolerance used when deciding whether a gap is a whole number of slices.
GAP_TOLERANCE = 0.15

PATIENT_AXES = (
    (0, 1.0, "Left (patient L)"),
    (0, -1.0, "Right (patient R)"),
    (1, 1.0, "Posterior (patient P)"),
    (1, -1.0, "Anterior (patient A)"),
    (2, 1.0, "Superior (patient S)"),
    (2, -1.0, "Inferior (patient I)"),
)


@dataclass
class OrderingResult:
    ordered: list[SliceGeometry]
    slice_positions_mm: np.ndarray
    slice_spacing_mm: float
    direction: np.ndarray  # (3,3)
    origin: np.ndarray  # (3,)
    row_direction: np.ndarray
    column_direction: np.ndarray
    slice_direction: np.ndarray
    orientation_label: str
    normal: np.ndarray
    irregular_spacing: bool
    spacing_std_mm: float
    max_deviation_mm: float
    duplicate_positions: int
    missing_slices: int
    gaps_detected: int
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    ordering_method: str = "spatial"

    @property
    def slice_count(self) -> int:
        return len(self.ordered)

    @property
    def determinant(self) -> float:
        return float(np.linalg.det(self.direction))


def dominant_axis_label(vector: np.ndarray) -> str:
    """Return a human label for the patient axis a direction cosine points along."""
    vec = np.asarray(vector, dtype=np.float64)
    idx = int(np.argmax(np.abs(vec)))
    sign = 1.0 if vec[idx] >= 0 else -1.0
    for axis, s, label in PATIENT_AXES:
        if axis == idx and abs(s - sign) < 1e-9:
            return label
    return "unknown"


def axis_name(vector: np.ndarray) -> str:
    """Compact axis name: ``+X``, ``-Y``, ... in LPS patient space."""
    vec = np.asarray(vector, dtype=np.float64)
    idx = int(np.argmax(np.abs(vec)))
    letter = "XYZ"[idx]
    return f"{'+' if vec[idx] >= 0 else '-'}{letter}"


def classify_orientation(row_dir: np.ndarray, col_dir: np.ndarray, normal: np.ndarray) -> tuple[str, float]:
    """Classify a slice stack as axial / coronal / sagittal / oblique.

    Returns the label and the obliquity in degrees (angle between the slice
    normal and the nearest patient axis).
    """
    n = np.asarray(normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) or 1.0)
    idx = int(np.argmax(np.abs(n)))
    axis_vectors = np.eye(3)
    best = axis_vectors[idx] * np.sign(n[idx])
    dot = float(np.clip(abs(np.dot(n, best)), 0.0, 1.0))
    obliquity = float(np.degrees(np.arccos(dot)))

    base = {0: "sagittal", 1: "coronal", 2: "axial"}[idx]
    if abs(n[idx]) >= AXIS_ALIGNED_COS:
        return base, obliquity
    return f"oblique ({base}-dominant)", obliquity


def _reference_orientation(
    geometries: list[SliceGeometry],
) -> tuple[np.ndarray, list[str]]:
    """Return a robust reference IOP for the series plus any warnings."""
    warnings: list[str] = []
    iops = np.array([g.image_orientation_patient[:6] for g in geometries], dtype=np.float64)
    reference = np.median(iops, axis=0)
    # Re-orthonormalise so numerically noisy headers cannot skew the basis.
    d_i = reference[:3]
    d_j = reference[3:]
    d_i = d_i / (np.linalg.norm(d_i) or 1.0)
    d_j = d_j - np.dot(d_j, d_i) * d_i
    d_j = d_j / (np.linalg.norm(d_j) or 1.0)
    reference = np.concatenate([d_i, d_j])

    deviations = np.linalg.norm(iops - reference[None, :], axis=1)
    inconsistent = int(np.count_nonzero(deviations > IOP_TOLERANCE))
    if inconsistent:
        warnings.append(
            f"{inconsistent} of {len(geometries)} slices report an ImageOrientationPatient "
            "that differs from the series reference. The median orientation was used; "
            "the reconstruction may be unreliable for these slices."
        )
    return reference, warnings


def _detect_gaps(sorted_positions: np.ndarray, spacing: float) -> tuple[int, int]:
    """Return ``(gap_count, missing_slice_count)`` for a sorted position array."""
    if sorted_positions.size < 2 or spacing <= 1e-9:
        return 0, 0
    diffs = np.diff(sorted_positions)
    ratios = diffs / spacing
    rounded = np.rint(ratios)
    gaps = 0
    missing = 0
    for ratio, whole in zip(ratios, rounded, strict=True):
        if whole >= 2 and abs(ratio - whole) <= GAP_TOLERANCE:
            gaps += 1
            missing += int(whole) - 1
    return gaps, missing


def _fallback_ordering(
    geometries: list[SliceGeometry],
    *,
    reason: str,
) -> tuple[list[SliceGeometry], np.ndarray, list[str]]:
    """Order by InstanceNumber when no spatial information is available.

    Only used when the caller explicitly opts in; the returned warnings always
    state loudly that the ordering is not spatially verified.
    """
    if all(g.instance_number is None for g in geometries):
        raise IncompleteOrientationError(
            "Unable to determine patient-space orientation.",
            detail=(
                "No ImagePositionPatient / ImageOrientationPatient values were present and no "
                "InstanceNumber could be used as a fallback. Slices cannot be ordered."
            ),
        )
    ordered = sorted(
        geometries,
        key=lambda g: (g.instance_number if g.instance_number is not None else 0),
    )
    thickness = None
    for g in ordered:
        thickness = g.spacing_between_slices or g.slice_thickness
        if thickness:
            break
    spacing = float(thickness or 1.0)
    positions = np.arange(len(ordered), dtype=np.float64) * spacing
    warnings = [
        reason,
        "FALLBACK ORDERING IN USE: slices were ordered by InstanceNumber because no "
        "spatial metadata was available. Anatomical position is NOT verified and "
        "measurements along the slice axis should be treated as unreliable.",
    ]
    return ordered, positions, warnings


def order_series(
    geometries: list[SliceGeometry],
    *,
    allow_instance_number_fallback: bool = False,
) -> OrderingResult:
    """Order slices physically and derive the reconstruction geometry."""
    if not geometries:
        raise DicomError("The selected series contains no readable image slices.")
    if len(geometries) == 1:
        geometries = list(geometries)

    usable: list[SliceGeometry] = []
    skipped = 0
    for geom in geometries:
        if geom.image_position_patient is None or geom.image_orientation_patient is None:
            skipped += 1
            continue
        if geom.image_orientation_patient.size < 6:
            skipped += 1
            continue
        usable.append(geom)

    warnings: list[str] = []
    errors: list[str] = []

    if skipped:
        warnings.append(
            f"{skipped} slice(s) were excluded because their spatial metadata is incomplete."
        )

    if not usable:
        reason = (
            "ImagePositionPatient and ImageOrientationPatient are missing for every slice in "
            "this series."
        )
        if not allow_instance_number_fallback:
            raise IncompleteOrientationError(
                "Unable to determine patient-space orientation.",
                detail=reason + " Enable the InstanceNumber fallback to attempt a non-spatial reconstruction.",
            )
        ordered, positions, fb_warnings = _fallback_ordering(geometries, reason=reason)
        warnings.extend(fb_warnings)
        d_i = np.array([1.0, 0.0, 0.0])
        d_j = np.array([0.0, 1.0, 0.0])
        d_k = np.array([0.0, 0.0, 1.0])
        return OrderingResult(
            ordered=ordered,
            slice_positions_mm=positions,
            slice_spacing_mm=float(np.median(np.diff(positions))) if positions.size > 1 else 1.0,
            direction=np.column_stack([d_k, d_j, d_i]),
            origin=np.asarray(ordered[0].image_position_patient, dtype=np.float64)
            if ordered[0].image_position_patient is not None
            else np.zeros(3),
            row_direction=d_j,
            column_direction=d_i,
            slice_direction=d_k,
            orientation_label="unknown (fallback ordering)",
            normal=d_k,
            irregular_spacing=False,
            spacing_std_mm=0.0,
            max_deviation_mm=0.0,
            duplicate_positions=0,
            missing_slices=0,
            gaps_detected=0,
            warnings=warnings,
            errors=errors,
            ordering_method="instance_number_fallback",
        )

    reference_iop, orientation_warnings = _reference_orientation(usable)
    warnings.extend(orientation_warnings)

    d_i = reference_iop[:3]
    d_j = reference_iop[3:]
    normal = np.cross(d_i, d_j)
    n_norm = np.linalg.norm(normal)
    if n_norm < 1e-9:
        raise IncompleteOrientationError(
            "Unable to determine patient-space orientation.",
            detail="ImageOrientationPatient direction cosines are degenerate (zero cross product).",
        )
    normal = normal / n_norm

    for geom in usable:
        geom.sort_key = float(np.dot(geom.image_position_patient, normal))

    ordered = sorted(usable, key=lambda g: g.sort_key)
    positions = np.array([g.sort_key for g in ordered], dtype=np.float64)

    # ---- duplicate detection -------------------------------------------
    duplicates = 0
    if positions.size > 1:
        diffs_all = np.diff(positions)
        duplicates = int(np.count_nonzero(np.abs(diffs_all) <= DUPLICATE_TOLERANCE_MM))
        if duplicates:
            warnings.append(
                f"{duplicates} duplicate slice position(s) detected: two or more files share the "
                "same patient-space location. One copy is retained in the volume."
            )
            keep = np.ones(positions.size, dtype=bool)
            keep[1:] = np.abs(np.diff(positions)) > DUPLICATE_TOLERANCE_MM
            ordered = [g for g, k in zip(ordered, keep, strict=True) if k]
            positions = positions[keep]

    # ---- spacing analysis ------------------------------------------------
    irregular = False
    spacing_std = 0.0
    max_dev = 0.0
    missing_slices = 0
    gaps = 0

    if positions.size > 1:
        diffs = np.diff(positions)
        spacing = float(np.median(diffs))
        if spacing <= 1e-9:
            raise DicomError(
                "Slice spacing could not be determined: multiple slices share the same "
                "patient-space position.",
            )
        spacing_std = float(np.std(diffs))
        max_dev = float(np.max(np.abs(diffs - spacing)))
        if max_dev / spacing > IRREGULARITY_TOLERANCE:
            irregular = True
            warnings.append(
                "Slice spacing is inconsistent: positions deviate by up to "
                f"{max_dev:.3f} mm from the median spacing of {spacing:.3f} mm "
                f"({100.0 * max_dev / spacing:.1f}%). The volume is resampled to a uniform grid."
            )
        gaps, missing_slices = _detect_gaps(positions, spacing)
        if missing_slices:
            warnings.append(
                f"{missing_slices} slice gap(s) detected across {gaps} location(s). "
                "The volume is resampled and gaps are interpolated; this may reduce accuracy "
                "along the slice axis."
            )
    else:
        thickness = None
        for geom in ordered:
            thickness = geom.spacing_between_slices or geom.slice_thickness
            if thickness:
                break
        spacing = float(thickness or 1.0)
        warnings.append(
            "Only a single slice is available in this series. Slice spacing was taken from the "
            f"DICOM header ({spacing:.3f} mm) and the reconstructed volume is one slice thick."
        )

    orientation_label, obliquity = classify_orientation(d_j, d_i, normal)
    if obliquity > 1.0 and "oblique" not in orientation_label:
        orientation_label = f"{orientation_label} (oblique {obliquity:.1f}°)"
    elif "oblique" in orientation_label:
        orientation_label = f"{orientation_label}, {obliquity:.1f}° from nearest axis"

    slice_direction = normal
    direction = np.column_stack([slice_direction, d_j, d_i])
    origin = np.asarray(ordered[0].image_position_patient, dtype=np.float64).copy()

    return OrderingResult(
        ordered=ordered,
        slice_positions_mm=positions,
        slice_spacing_mm=float(spacing),
        direction=direction,
        origin=origin,
        row_direction=d_j,
        column_direction=d_i,
        slice_direction=slice_direction,
        orientation_label=orientation_label,
        normal=normal,
        irregular_spacing=irregular,
        spacing_std_mm=spacing_std,
        max_deviation_mm=max_dev,
        duplicate_positions=duplicates,
        missing_slices=missing_slices,
        gaps_detected=gaps,
        warnings=warnings,
        errors=errors,
        ordering_method="spatial",
    )


def describe_frame(ordering: OrderingResult) -> dict[str, str]:
    """Human-readable description of the patient-space frame for the UI."""
    return {
        "slice_axis": axis_name(ordering.slice_direction),
        "row_axis": axis_name(ordering.row_direction),
        "column_axis": axis_name(ordering.column_direction),
        "slice_axis_patient": dominant_axis_label(ordering.slice_direction),
        "row_axis_patient": dominant_axis_label(ordering.row_direction),
        "column_axis_patient": dominant_axis_label(ordering.column_direction),
    }
