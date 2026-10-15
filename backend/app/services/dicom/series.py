"""Series detection and dataset-level analysis.

Given a scanned folder this module:

1. reads the header of every file that parsed as DICOM,
2. extracts the metadata subset we care about,
3. groups slices into series by ``SeriesInstanceUID``,
4. groups series into studies by ``StudyInstanceUID``,
5. derives per-series geometry (orientation, spacing, completeness).

No patient-identifying attributes are ever extracted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...core.errors import DicomError, NoValidDicomSeriesError
from ...core.logging_ import get_logger
from ...models import DatasetRef, DicomMetadata, SeriesInfo, SliceGeometry
from . import reader
from .series_ordering import classify_orientation, order_series, describe_frame

logger = get_logger("dicom.series")

#: Modalities we can reconstruct a volume from.
VOLUMETRIC_MODALITIES = {"CT", "MR", "PT", "NM", "CBCT", "US"}

#: Modalities we explicitly label as unsupported for segmentation presets.
HU_MODALITIES = {"CT", "CBCT"}


@dataclass
class SeriesFile:
    """One image file belonging to a series."""

    rel_path: str
    original_name: str
    geometry: SliceGeometry | None
    meta: DicomMetadata
    frames: int = 1
    frame_index: int = 0


@dataclass
class SeriesAnalysis:
    info: SeriesInfo
    files: list[SeriesFile]
    ordering_warnings: list[str] = field(default_factory=list)
    frame_description: dict[str, str] = field(default_factory=dict)


@dataclass
class DatasetAnalysis:
    dataset_id: str
    studies: list[dict[str, object]]
    series: list[SeriesAnalysis]
    non_image_files: int
    unreadable_files: int
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    primary_series_uid: str | None = None

    def series_by_uid(self, uid: str) -> SeriesAnalysis | None:
        for entry in self.series:
            if entry.info.series_instance_uid == uid:
                return entry
        return None


def _majority(values: list[str | None]) -> str | None:
    counts: dict[str, int] = {}
    for value in values:
        if value:
            counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _median_value(values: list[float | None]) -> float | None:
    usable = [v for v in values if v is not None and np.isfinite(v)]
    if not usable:
        return None
    return float(np.median(usable))


def analyze_dataset(
    dataset: DatasetRef,
    root: Path,
    *,
    progress=None,
) -> DatasetAnalysis:
    """Read headers, group into series, and produce per-series geometry."""
    started = time.time()
    warnings: list[str] = []
    errors: list[str] = []

    dicom_records = dataset.dicom_files
    if not dicom_records:
        raise NoValidDicomSeriesError(
            "Uploaded files do not contain a valid DICOM series.",
            detail=(
                f"Scanned {dataset.file_count} file(s); none could be parsed as DICOM. "
                "Check that the folder contains the original DICOM slices."
            ),
        )

    if progress:
        progress(f"{len(dicom_records)} DICOM file(s) found; reading headers")

    groups: dict[str, list[SeriesFile]] = {}
    non_image = 0
    unreadable = 0
    total = len(dicom_records)

    for index, record in enumerate(dicom_records):
        path = root / record.path
        try:
            ds = reader.read_dataset(path, stop_before_pixels=True)
            meta = reader.extract_metadata(ds)
        except DicomError as exc:
            unreadable += 1
            record.error = exc.message
            continue
        except Exception as exc:  # noqa: BLE001
            unreadable += 1
            record.error = f"Header could not be read ({type(exc).__name__})."
            continue

        if meta.rows is None or meta.columns is None:
            non_image += 1
            record.error = "Not an image object (no Rows/Columns)."
            continue

        try:
            frames = max(1, int(ds.get("NumberOfFrames", 1)))
        except (TypeError, ValueError):
            frames = 1

        geometry = reader.parse_slice_geometry(ds, meta)
        series_uid = meta.series_instance_uid or f"unknown-series-{meta.study_instance_uid or 'x'}"
        bucket = groups.setdefault(series_uid, [])
        if frames > 1:
            # Multi-frame objects: each frame is an independent slice.  Per-frame
            # positions are derived from SpacingBetweenSlices/SliceThickness when
            # present; a warning is emitted because enhanced multi-frame objects
            # technically carry per-frame positions in PerFrameFunctionalGroupsSequence.
            step = meta.spacing_between_slices or meta.slice_thickness
            if not step:
                dataset.warnings.append(
                    f"'{record.original_name}' is a multi-frame object with no "
                    "SpacingBetweenSlices/SliceThickness, so its frames could not be positioned."
                )
            for frame in range(frames):
                bucket.append(
                    SeriesFile(
                        rel_path=record.path,
                        original_name=record.original_name,
                        geometry=reader.parse_slice_geometry(ds, meta, frame_index=frame),
                        meta=meta,
                        frames=frames,
                        frame_index=frame,
                    )
                )
        else:
            bucket.append(
                SeriesFile(
                    rel_path=record.path,
                    original_name=record.original_name,
                    geometry=geometry,
                    meta=meta,
                    frames=1,
                    frame_index=0,
                )
            )

        if progress and (index % 100 == 0 or index == total - 1):
            progress(f"Read metadata for {index + 1}/{total} file(s)")

    if not groups:
        raise NoValidDicomSeriesError(
            "Uploaded files do not contain a valid DICOM image series.",
            detail=f"{non_image} file(s) parsed as DICOM but carried no image pixel data.",
        )

    analyses: list[SeriesAnalysis] = []
    for series_uid, files in groups.items():
        analyses.append(_build_series(series_uid, files))

    # Most useful series first: largest slice count, then lowest series number.
    analyses.sort(
        key=lambda a: (-a.info.slice_count, a.info.series_number if a.info.series_number is not None else 10**6)
    )

    studies: dict[str, list[SeriesAnalysis]] = {}
    for entry in analyses:
        key = entry.info.study_instance_uid or "unknown-study"
        studies.setdefault(key, []).append(entry)

    study_summaries: list[dict[str, object]] = []
    for study_uid, entries in studies.items():
        study_summaries.append(
            {
                "study_instance_uid": study_uid,
                "series_count": len(entries),
                "slice_count": int(sum(e.info.slice_count for e in entries)),
                "study_description": _majority([e.info.study_description for e in entries]),
                "modalities": sorted({e.info.modality for e in entries if e.info.modality}),
                "series_instance_uids": [e.info.series_instance_uid for e in entries],
            }
        )
    study_summaries.sort(key=lambda s: -int(s["slice_count"]))

    if len(study_summaries) > 1:
        warnings.append(
            f"{len(study_summaries)} studies were found in this upload. The study with the most "
            "slices is pre-selected; choose a specific series to reconstruct a different one."
        )
    if len(analyses) > 1:
        warnings.append(
            f"Multiple DICOM series detected ({len(analyses)}). Select the series you want to "
            "reconstruct -- mixing series is not supported."
        )
    if unreadable:
        warnings.append(f"{unreadable} DICOM file(s) could not have their header read and were ignored.")
    if non_image:
        warnings.append(f"{non_image} DICOM object(s) contained no image data and were ignored.")

    return DatasetAnalysis(
        dataset_id=dataset.dataset_id,
        studies=study_summaries,
        series=analyses,
        non_image_files=non_image,
        unreadable_files=unreadable,
        warnings=warnings,
        errors=errors,
        elapsed_s=time.time() - started,
        primary_series_uid=analyses[0].info.series_instance_uid if analyses else None,
    )


def _build_series(series_uid: str, files: list[SeriesFile]) -> SeriesAnalysis:
    """Summarise one series and pre-compute its geometry."""
    geometries = [f.geometry for f in files if f.geometry is not None]
    modalities = [f.meta.modality for f in files]
    modality = _majority(modalities) or "OT"

    info = SeriesInfo(
        series_instance_uid=series_uid,
        study_instance_uid=_majority([f.meta.study_instance_uid for f in files]),
        modality=modality,
        series_number=next((f.meta.series_number for f in files if f.meta.series_number is not None), None),
        series_description=_majority([f.meta.series_description for f in files]),
        study_description=_majority([f.meta.study_description for f in files]),
        body_part_examined=_majority([f.meta.body_part_examined for f in files]),
        manufacturer=_majority([f.meta.manufacturer for f in files]),
        rows=next((f.meta.rows for f in files if f.meta.rows), None),
        columns=next((f.meta.columns for f in files if f.meta.columns), None),
        slice_count=len(files),
        files_with_pixel_data=len(files),
    )

    warnings: list[str] = []
    frame_description: dict[str, str] = {}

    if geometries:
        try:
            ordering = order_series(geometries)
        except DicomError as exc:
            info.issues.append(exc.message)
            warnings.append(exc.message)
        else:
            info.orientation_label = ordering.orientation_label
            info.normal = ordering.normal
            info.slice_spacing = ordering.slice_spacing_mm
            info.irregular_spacing = ordering.irregular_spacing
            info.has_complete_geometry = True
            warnings.extend(ordering.warnings)
            frame_description = describe_frame(ordering)
            if ordering.ordering_method != "spatial":
                info.has_complete_geometry = False

    pixel_spacings = [f.meta.pixel_spacing for f in files if f.meta.pixel_spacing]
    if pixel_spacings:
        first = pixel_spacings[0]
        mismatched = sum(1 for p in pixel_spacings if abs(p[0] - first[0]) > 1e-6 or abs(p[1] - first[1]) > 1e-6)
        if mismatched:
            warnings.append(
                f"{mismatched} slice(s) report a different PixelSpacing from the first slice."
            )

    missing_spatial = [f for f in files if f.meta.image_position_patient is None or f.meta.image_orientation_patient is None]
    if missing_spatial:
        info.has_complete_geometry = False
        warnings.append(
            f"{len(missing_spatial)} of {len(files)} slice(s) in this series are missing "
            "ImagePositionPatient or ImageOrientationPatient."
        )

    if modality not in VOLUMETRIC_MODALITIES:
        warnings.append(
            f"Modality '{modality}' is not a standard volumetric modality; reconstruction may not be "
            "meaningful."
        )

    if info.rows and info.columns:
        shapes = {(f.meta.rows, f.meta.columns) for f in files}
        if len(shapes) > 1:
            warnings.append(
                f"Inconsistent in-plane dimensions within this series ({len(shapes)} distinct "
                "Rows x Columns combinations)."
            )

    return SeriesAnalysis(
        info=info,
        files=files,
        ordering_warnings=warnings,
        frame_description=frame_description,
    )


def series_orientation_preview(analysis: SeriesAnalysis) -> tuple[str, float]:
    """Convenience wrapper used by the report builder."""
    geometries = [f.geometry for f in analysis.files if f.geometry is not None]
    if not geometries:
        return ("unknown", 0.0)
    iop = np.median(np.array([g.image_orientation_patient[:6] for g in geometries]), axis=0)
    d_i, d_j = iop[:3], iop[3:]
    normal = np.cross(d_i, d_j)
    return classify_orientation(d_j, d_i, normal)


def modality_supports_hu(modality: str | None) -> bool:
    return (modality or "").upper() in HU_MODALITIES
