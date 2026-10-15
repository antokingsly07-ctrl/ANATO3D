"""3D voxel volume reconstruction from ordered DICOM slices.

The reconstruction is deliberately explicit about the patient-space frame it
produces:

``physical_lps = origin + direction @ (index * spacing)``

with ``index = (k, j, i)``:  ``k`` along the slice axis, ``j`` along rows,
``i`` along columns.  ``direction`` has columns ``[d_slice, d_row, d_column]``
as derived in :mod:`app.services.dicom.series_ordering`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...core.errors import ReconstructionError
from ...core.logging_ import get_logger
from ...core.storage import new_id
from ...models import DatasetRef, Volume
from ..dicom import reader
from ..dicom.series import DatasetAnalysis, SeriesAnalysis, SeriesFile, modality_supports_hu
from ..dicom.series_ordering import OrderingResult, order_series

logger = get_logger("reconstruction.volume")

#: Above this many voxels we prefer int16 storage when the values allow it.
COMPACT_DTYPE_THRESHOLD = 60_000_000
#: Refuse to allocate more than this many bytes for the source volume.
MAX_VOLUME_BYTES = 3_000_000_000
#: Rows processed per interpolation block (bounds peak memory during resampling).
RESAMPLE_BLOCK_ROWS = 64


@dataclass
class ReconstructionResult:
    volume: Volume
    ordering: OrderingResult
    report: dict[str, object]
    frames: dict[str, object] = field(default_factory=dict)


def _pick_dtype(
    dataset: DatasetRef,
    root: Path,
    ordered_files: list[SeriesFile],
    n_voxels: int,
    *,
    progress=None,
) -> np.dtype:
    """Choose int16 or float32 storage, verifying the value range first."""
    if n_voxels <= COMPACT_DTYPE_THRESHOLD:
        return np.dtype(np.float32)

    sample_count = min(6, len(ordered_files))
    indices = np.linspace(0, len(ordered_files) - 1, sample_count).astype(int)
    vmin = np.inf
    vmax = -np.inf
    for index in indices:
        try:
            frames, _ = reader.load_pixel_frames(root / ordered_files[int(index)].rel_path)
        except Exception:  # noqa: BLE001 - fall back to float32 on any read issue
            return np.dtype(np.float32)
        vmin = min(vmin, float(frames.min()))
        vmax = max(vmax, float(frames.max()))

    fits_int16 = vmin >= -32768.0 and vmax <= 32767.0
    if fits_int16:
        if progress:
            progress.log(
                f"Large volume ({n_voxels / 1e6:.1f}M voxels): using compact int16 storage "
                f"(value range {vmin:.0f}..{vmax:.0f} fits exactly)."
            )
        return np.dtype(np.int16)
    return np.dtype(np.float32)


def _interpolate_axis0(array: np.ndarray, src_positions: np.ndarray, dst_positions: np.ndarray) -> np.ndarray:
    """Linear interpolation along axis 0 at arbitrary destination positions."""
    src = np.asarray(src_positions, dtype=np.float64)
    dst = np.asarray(dst_positions, dtype=np.float64)
    if src.size == 1:
        return np.repeat(array, dst.size, axis=0).astype(array.dtype, copy=True)

    index = np.searchsorted(src, dst, side="right") - 1
    index = np.clip(index, 0, src.size - 2)
    denom = src[index + 1] - src[index]
    denom[denom == 0] = 1.0
    weight = (dst - src[index]) / denom
    weight = np.clip(weight, 0.0, 1.0)

    out = np.empty((dst.size,) + array.shape[1:], dtype=np.float32)
    rows = array.shape[1]
    step = max(1, RESAMPLE_BLOCK_ROWS)
    for start in range(0, rows, step):
        stop = min(start + step, rows)
        block = array[:, start:stop, :]
        lo = block[index]
        hi = block[index + 1]
        out[:, start:stop, :] = (
            lo.astype(np.float32) * (1.0 - weight)[:, None, None]
            + hi.astype(np.float32) * weight[:, None, None]
        )
    return out


def resample_volume(
    array: np.ndarray,
    spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Resample onto a uniform grid with ``target_spacing``.

    The direction matrix and origin are unchanged: resampling only changes the
    sampling density along the array axes, not the patient-space frame.
    """
    result = array
    current = list(spacing)
    for axis in range(3):
        if abs(current[axis] - target_spacing[axis]) < 1e-6:
            continue
        moved = np.moveaxis(result, axis, 0)
        count = moved.shape[0]
        src_positions = np.arange(count, dtype=np.float64) * current[axis]
        span = (count - 1) * current[axis]
        new_count = max(1, int(round(span / target_spacing[axis])) + 1)
        dst_positions = np.arange(new_count, dtype=np.float64) * target_spacing[axis]
        interpolated = _interpolate_axis0(moved, src_positions, dst_positions)
        result = np.moveaxis(interpolated, 0, axis)
        current[axis] = target_spacing[axis]
    return result, (current[0], current[1], current[2])


def _resolve_target_spacing(
    ordering: OrderingResult,
    row_spacing: float,
    column_spacing: float,
    *,
    uniform_spacing_mm: tuple[float, float, float] | None,
    isotropic: bool,
) -> tuple[float, float, float]:
    base = np.array(
        [ordering.slice_spacing_mm, row_spacing, column_spacing],
        dtype=np.float64,
    )
    if uniform_spacing_mm is not None:
        return (float(uniform_spacing_mm[0]), float(uniform_spacing_mm[1]), float(uniform_spacing_mm[2]))
    if isotropic:
        target = float(np.min(base[base > 0])) if np.any(base > 0) else 1.0
        return (target, target, target)
    return (float(base[0]), float(base[1]), float(base[2]))


def build_volume(
    dataset: DatasetRef,
    root: Path,
    series: SeriesAnalysis,
    *,
    allow_instance_number_fallback: bool = False,
    uniform_spacing_mm: tuple[float, float, float] | None = None,
    isotropic: bool = False,
    progress=None,
) -> ReconstructionResult:
    """Stack an ordered slice series into a patient-space volume."""
    started = time.time()
    if progress:
        progress.step("slice_ordering", 0.02, f"Ordering {len(series.files)} slice(s) by patient position")

    file_by_geometry: dict[int, SeriesFile] = {}
    geometries = []
    for entry in series.files:
        if entry.geometry is None:
            continue
        geometries.append(entry.geometry)
        file_by_geometry[id(entry.geometry)] = entry

    ordering = order_series(
        geometries,
        allow_instance_number_fallback=allow_instance_number_fallback,
    )

    ordered_files = [file_by_geometry[id(g)] for g in ordering.ordered]

    rows = ordered_files[0].meta.rows
    columns = ordered_files[0].meta.columns
    if rows is None or columns is None:
        raise ReconstructionError("Slice dimensions are missing from the DICOM headers.")

    for entry in ordered_files:
        if entry.meta.rows != rows or entry.meta.columns != columns:
            raise ReconstructionError(
                "Slices in this series have inconsistent in-plane dimensions "
                f"({entry.meta.rows}x{entry.meta.columns} vs {rows}x{columns}). "
                "Reconstruct each sub-series separately.",
                detail=f"Offending file: {entry.original_name}",
            )

    depth = len(ordered_files)
    n_voxels = depth * rows * columns
    itemsize = 4
    estimated = n_voxels * itemsize
    if estimated > MAX_VOLUME_BYTES:
        raise ReconstructionError(
            f"The reconstructed volume would need approximately {estimated / 1e9:.1f} GB "
            f"({depth} x {rows} x {columns} voxels), which exceeds the configured limit.",
            detail="Select a smaller series or increase ANATO3D_MAX_VOXELS_HARD in the configuration.",
        )

    dtype = _pick_dtype(dataset, root, ordered_files, n_voxels, progress=progress)

    if progress:
        progress.step(
            "volume_reconstruction",
            0.05,
            f"Allocating {depth} x {rows} x {columns} {dtype.name} volume",
        )

    volume_array = np.zeros((depth, rows, columns), dtype=dtype)

    cached_path: Path | None = None
    cached_frames: np.ndarray | None = None
    raw_units: set[str] = set()

    for index, entry in enumerate(ordered_files):
        path = root / entry.rel_path
        try:
            if cached_path != path or cached_frames is None:
                loaded, _ = reader.load_pixel_frames(path)
                cached_frames = loaded
                cached_path = path
        except Exception as exc:  # noqa: BLE001
            raise ReconstructionError(
                f"Pixel data could not be read for '{entry.original_name}'.",
                detail=str(exc),
            ) from exc

        frame = cached_frames[min(entry.frame_index, cached_frames.shape[0] - 1)]
        if frame.shape != (rows, columns):
            raise ReconstructionError(
                "A slice in this series does not match the expected dimensions.",
                detail=f"'{entry.original_name}' is {frame.shape[0]}x{frame.shape[1]}, expected {rows}x{columns}.",
            )
        volume_array[index] = np.rint(frame) if dtype == np.int16 else frame
        raw_units.add(entry.meta.modality or "OT")

        if progress and (index % 25 == 0 or index == depth - 1):
            progress.step(
                "volume_reconstruction",
                0.05 + 0.55 * (index + 1) / depth,
                f"Loaded slice {index + 1}/{depth}",
            )

    modality = series.info.modality
    intensity_units = "HU" if modality_supports_hu(modality) else "intensity"
    if modality_supports_hu(modality):
        has_rescale = any(
            (f.meta.rescale_slope not in (None, 1) or f.meta.rescale_intercept not in (None, 0))
            for f in ordered_files
        )
        if not has_rescale:
            series.ordering_warnings.append(
                "CT series has no RescaleSlope/RescaleIntercept; raw stored values are used as "
                "Hounsfield-equivalent values."
            )

    row_spacing = float(np.median([f.geometry.row_spacing for f in ordered_files]))  # type: ignore[union-attr]
    column_spacing = float(np.median([f.geometry.column_spacing for f in ordered_files]))  # type: ignore[union-attr]

    window_center = _series_window(ordered_files, "window_center") or _series_window(ordered_files, "default_window_center")
    window_width = _series_window(ordered_files, "window_width") or _series_window(ordered_files, "default_window_width")
    if modality_supports_hu(modality) and (window_center is None or window_width is None):
        window_center, window_width = 40.0, 400.0

    volume = Volume(
        volume_id=new_id("vol"),
        array=volume_array,
        spacing=(ordering.slice_spacing_mm, row_spacing, column_spacing),
        origin=ordering.origin,
        direction=ordering.direction,
        modality=modality,
        series_instance_uid=series.info.series_instance_uid,
        study_instance_uid=series.info.study_instance_uid,
        series_description=series.info.series_description,
        patient_position=_series_patient_position(ordered_files),
        orientation_label=ordering.orientation_label,
        window_center=window_center,
        window_width=window_width,
        slice_positions_mm=ordering.slice_positions_mm,
        source_id=dataset.dataset_id,
        intensity_units=intensity_units,
    )

    # ---- resampling -----------------------------------------------------
    target = _resolve_target_spacing(
        ordering,
        row_spacing,
        column_spacing,
        uniform_spacing_mm=uniform_spacing_mm,
        isotropic=isotropic,
    )
    ideal_depth = depth
    current = list(volume.spacing)
    needs_resample = (
        ordering.irregular_spacing
        or ordering.missing_slices > 0
        or any(abs(current[a] - target[a]) > 1e-4 for a in range(3))
    )

    if needs_resample:
        if progress:
            progress.step(
                "preprocessing",
                0.62,
                "Resampling to a uniform {:.3f} x {:.3f} x {:.3f} mm grid".format(*target),
            )
        positions = ordering.slice_positions_mm
        resampled, new_spacing = resample_volume(volume_array, volume.spacing, target)
        volume = volume.copy_with(
            array=resampled,
            spacing=new_spacing,
            is_resampled=True,
        )
        ideal_depth = resampled.shape[0]
        volume.notes.append(
            "Volume resampled to a uniform {:.3f} x {:.3f} x {:.3f} mm voxel grid.".format(*new_spacing)
        )
        volume.warnings.extend(ordering.warnings)
        del positions

    if ordering.irregular_spacing or ordering.missing_slices:
        volume.warnings.append(
            "Source slices were not uniformly spaced; the volume was resampled and interpolated. "
            "Measurements along the slice axis carry additional uncertainty."
        )

    volume.notes.append(
        "Patient-space frame: origin=[{:.3f}, {:.3f}, {:.3f}] mm (LPS), orientation {}".format(
            volume.origin[0], volume.origin[1], volume.origin[2], ordering.orientation_label
        )
    )

    report = build_reconstruction_report(volume, ordering, series, depth, ideal_depth)
    report["elapsed_s"] = round(time.time() - started, 3)
    report["source_files"] = len(ordered_files)

    if progress:
        progress.step("preprocessing", 0.72, "Volume reconstruction complete")

    return ReconstructionResult(
        volume=volume,
        ordering=ordering,
        report=report,
        frames={
            "orientation": ordering.orientation_label,
            "slice_axis": ordering.slice_direction.tolist(),
        },
    )


def _series_window(files: list[SeriesFile], attribute: str) -> float | None:
    values = []
    for entry in files:
        value = getattr(entry.meta, attribute, None)
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    return float(np.median(values))


def _series_patient_position(files: list[SeriesFile]) -> str | None:
    for entry in files:
        if entry.meta.patient_position:
            return entry.meta.patient_position
    return None


def build_reconstruction_report(
    volume: Volume,
    ordering: OrderingResult,
    series: SeriesAnalysis,
    source_slice_count: int,
    reconstructed_slice_count: int,
) -> dict[str, object]:
    return {
        "modality": volume.modality,
        "series_instance_uid": volume.series_instance_uid,
        "series_description": volume.series_description,
        "dimensions": list(volume.shape),
        "dimensions_order": "slices x rows x columns",
        "voxel_spacing_mm": [round(float(v), 6) for v in volume.spacing],
        "voxel_spacing_order": "slice x row x column",
        "voxel_volume_mm3": round(volume.voxel_volume_mm3, 6),
        "physical_size_mm": [round(float(v), 3) for v in volume.physical_size_mm],
        "origin_lps_mm": [round(float(v), 4) for v in volume.origin],
        "direction_matrix": np.asarray(volume.direction).round(6).tolist(),
        "direction_determinant": round(ordering.determinant, 6),
        "orientation": ordering.orientation_label,
        "patient_position": volume.patient_position,
        "slice_count": source_slice_count,
        "reconstructed_slice_count": reconstructed_slice_count,
        "irregular_spacing": bool(ordering.irregular_spacing),
        "slice_spacing_std_mm": round(float(ordering.spacing_std_mm), 4),
        "max_spacing_deviation_mm": round(float(ordering.max_deviation_mm), 4),
        "duplicate_slices": int(ordering.duplicate_positions),
        "missing_slices": int(ordering.missing_slices),
        "resampled": bool(volume.is_resampled),
        "normalized": bool(volume.is_normalized),
        "intensity_units": volume.intensity_units,
        "ordering_method": ordering.ordering_method,
        "window_center": volume.window_center,
        "window_width": volume.window_width,
        "orientation_frame": {
            "slice_axis": ordering.slice_direction.tolist(),
            "row_axis": ordering.row_direction.tolist(),
            "column_axis": ordering.column_direction.tolist(),
        },
        "warnings": list(ordering.warnings) + list(volume.warnings),
    }


def available_series(dataset_analysis: DatasetAnalysis) -> list[dict[str, object]]:
    """Compact series summaries for the UI series selector."""
    out = []
    for entry in dataset_analysis.series:
        info = entry.info
        out.append(
            {
                "series_instance_uid": info.series_instance_uid,
                "study_instance_uid": info.study_instance_uid,
                "modality": info.modality,
                "series_number": info.series_number,
                "series_description": info.series_description,
                "body_part_examined": info.body_part_examined,
                "rows": info.rows,
                "columns": info.columns,
                "slice_count": info.slice_count,
                "orientation": info.orientation_label,
                "slice_spacing_mm": round(info.slice_spacing, 4) if info.slice_spacing else None,
                "irregular_spacing": info.irregular_spacing,
                "has_complete_geometry": info.has_complete_geometry,
                "is_reconstructable": bool(info.has_complete_geometry and info.slice_count > 0),
                "issues": list(info.issues),
                "warnings": list(entry.ordering_warnings),
            }
        )
    return out
