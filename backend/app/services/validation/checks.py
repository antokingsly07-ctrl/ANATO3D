"""Automatic consistency checks over a dataset, reconstruction and segmentation.

Each finding carries a severity level so the UI can present them distinctly:

``error``    processing cannot be trusted; the user must intervene
``warning``  processing continued but numbers may be less reliable
``info``     informational, including "we changed something for you"

Findings never contain patient-identifying information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...core.logging_ import get_logger
from ...models import DatasetRef, SegmentationResult, SeriesInfo, Volume
from ..dicom.series import DatasetAnalysis
from ..dicom.series_ordering import OrderingResult

logger = get_logger("validation.checks")

#: Below this slice spacing (mm) a volume is suspiciously finely sampled.
MIN_PLAUSIBLE_SPACING_MM = 0.05
#: Above this slice spacing (mm) a volume is suspiciously coarsely sampled.
MAX_PLAUSIBLE_SLICE_SPACING_MM = 15.0
#: Above this in-plane spacing (mm) a volume is suspiciously coarse.
MAX_PLAUSIBLE_PIXEL_SPACING_MM = 8.0
#: Above this voxel diagonal (mm) physical detail is very limited.
COARSE_VOXEL_DIAGONAL_MM = 12.0


@dataclass
class Finding:
    code: str
    severity: str  # error | warning | info
    title: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "context": self.context,
        }


def _finding(code: str, severity: str, title: str, message: str, **context: Any) -> Finding:
    return Finding(code=code, severity=severity, title=title, message=message, context=context)


def check_dataset(dataset: DatasetRef, analysis: DatasetAnalysis | None = None) -> list[Finding]:
    findings: list[Finding] = []

    if not dataset.dicom_files:
        findings.append(
            _finding(
                "no_dicom",
                "error",
                "No valid DICOM files",
                "Uploaded files do not contain a valid DICOM series.",
                file_count=dataset.file_count,
            )
        )
        return findings

    unreadable = [f for f in dataset.dicom_files if f.error and "no image pixel data" not in (f.error or "")]
    if unreadable:
        findings.append(
            _finding(
                "invalid_dicom",
                "warning",
                "Unreadable DICOM files",
                f"{len(unreadable)} DICOM file(s) could not be parsed and were ignored.",
                examples=[f.original_name for f in unreadable[:5]],
            )
        )

    if analysis is not None:
        if len(analysis.studies) > 1:
            findings.append(
                _finding(
                    "multiple_studies",
                    "warning",
                    "Multiple studies detected",
                    f"{len(analysis.studies)} different StudyInstanceUIDs were found in this upload. "
                    "Reconstruct one study at a time.",
                    study_count=len(analysis.studies),
                )
            )
        if len(analysis.series) > 1:
            findings.append(
                _finding(
                    "multiple_series",
                    "info",
                    "Multiple series detected",
                    f"{len(analysis.series)} series are present. Only the selected series is reconstructed.",
                    series_count=len(analysis.series),
                )
            )
        if analysis.unreadable_files:
            findings.append(
                _finding(
                    "unreadable_headers",
                    "warning",
                    "Unreadable headers",
                    f"{analysis.unreadable_files} file(s) could not have their header read.",
                )
            )

    return findings


def check_series(info: SeriesInfo) -> list[Finding]:
    findings: list[Finding] = []

    if not info.has_complete_geometry:
        findings.append(
            _finding(
                "incomplete_geometry",
                "error",
                "Incomplete spatial metadata",
                "Image orientation metadata is incomplete for this series, so a physically correct "
                "reconstruction cannot be guaranteed.",
                series_instance_uid=info.series_instance_uid,
            )
        )
    if info.irregular_spacing:
        findings.append(
            _finding(
                "irregular_spacing",
                "warning",
                "Slice spacing is inconsistent",
                "Slice spacing is inconsistent. The volume is resampled to a uniform grid before "
                "segmentation; measurements along the slice axis carry extra uncertainty.",
                modality=info.modality,
                slice_spacing_mm=info.slice_spacing,
            )
        )
    if info.slice_spacing is not None and info.slice_spacing > MAX_PLAUSIBLE_SLICE_SPACING_MM:
        findings.append(
            _finding(
                "coarse_slice_spacing",
                "warning",
                "Unusually large slice spacing",
                f"Slice spacing is {info.slice_spacing:.2f} mm, which is coarse for volumetric "
                "reconstruction. Structures thinner than this may be missed.",
                slice_spacing_mm=info.slice_spacing,
            )
        )
    if info.slice_spacing is not None and 0 < info.slice_spacing < MIN_PLAUSIBLE_SPACING_MM:
        findings.append(
            _finding(
                "implausible_slice_spacing",
                "error",
                "Implausibly small slice spacing",
                f"Slice spacing resolved to {info.slice_spacing:.4f} mm. This usually indicates "
                "duplicated slices or corrupt spatial metadata.",
                slice_spacing_mm=info.slice_spacing,
            )
        )
    modality = (info.modality or "").upper()
    if modality not in {"CT", "MR", "PT", "NM", "CBCT"}:
        findings.append(
            _finding(
                "unsupported_modality",
                "warning",
                "Modality not designed for volumetric reconstruction",
                f"Modality '{info.modality}' is not a standard volumetric modality. Segmentation "
                "presets are generic for this modality; Hounsfield-unit presets do not apply.",
                modality=info.modality,
            )
        )
    if info.rows and info.columns:
        if info.rows < 16 or info.columns < 16:
            findings.append(
                _finding(
                    "unexpected_dimensions",
                    "warning",
                    "Unexpectedly small images",
                    f"Slice dimensions are {info.rows} x {info.columns}; this is much smaller than a "
                    "typical reconstructed CT/MR slice.",
                    rows=info.rows,
                    columns=info.columns,
                )
            )
        if info.rows * info.columns > 64_000_000:
            findings.append(
                _finding(
                    "huge_dimensions",
                    "warning",
                    "Very large slices",
                    f"Slice dimensions are {info.rows} x {info.columns}, which may exceed available memory.",
                    rows=info.rows,
                    columns=info.columns,
                )
            )
    return findings


def check_ordering(ordering: OrderingResult, info: SeriesInfo) -> list[Finding]:
    findings: list[Finding] = []

    if ordering.ordering_method != "spatial":
        findings.append(
            _finding(
                "non_spatial_ordering",
                "error",
                "Non-spatial slice ordering",
                "Slices were ordered by InstanceNumber because patient-space geometry was "
                "unavailable. Anatomical ordering is NOT verified.",
                method=ordering.ordering_method,
            )
        )
    if ordering.duplicate_positions:
        findings.append(
            _finding(
                "duplicate_slices",
                "warning",
                "Duplicate slice positions",
                f"{ordering.duplicate_positions} slice(s) shared an identical patient-space "
                "position; duplicates were removed from the volume.",
                count=ordering.duplicate_positions,
            )
        )
    if ordering.missing_slices:
        findings.append(
            _finding(
                "missing_slices",
                "warning",
                "Gaps between slices",
                f"{ordering.missing_slices} missing slice position(s) were inferred from the spacing "
                "pattern. The volume was resampled and the gaps interpolated.",
                missing_slices=ordering.missing_slices,
                gaps=ordering.gaps_detected,
            )
        )
    if ordering.determinant > 0:
        findings.append(
            _finding(
                "unexpected_handedness",
                "info",
                "Unexpected orientation handedness",
                "The direction matrix determinant is positive; this is unusual for a standard DICOM "
                "series and was compensated for when generating the surface.",
                determinant=round(ordering.determinant, 6),
            )
        )
    for message in ordering.warnings:
        if "differs from the series reference" in message:
            findings.append(
                _finding("inconsistent_orientation", "warning", "Inconsistent slice orientation", message)
            )
    return findings


def check_volume(volume: Volume, *, mask: np.ndarray | None = None) -> list[Finding]:
    findings: list[Finding] = []
    spacing = np.asarray(volume.spacing, dtype=np.float64)

    if np.any(spacing <= 1e-6):
        findings.append(
            _finding(
                "zero_spacing",
                "error",
                "Invalid voxel spacing",
                "At least one voxel spacing component is zero, so physical measurements cannot be computed.",
                spacing_mm=[float(v) for v in spacing],
            )
        )
    if float(np.linalg.norm(spacing)) > COARSE_VOXEL_DIAGONAL_MM:
        findings.append(
            _finding(
                "coarse_voxels",
                "warning",
                "Very coarse voxels",
                f"The voxel diagonal is {float(np.linalg.norm(spacing)):.2f} mm; fine anatomical detail "
                "cannot be represented at this resolution.",
                spacing_mm=[float(v) for v in spacing],
            )
        )
    if np.max(spacing[:2]) > MAX_PLAUSIBLE_PIXEL_SPACING_MM:
        findings.append(
            _finding(
                "coarse_in_plane",
                "warning",
                "Unusually large in-plane pixel spacing",
                f"In-plane pixel spacing is up to {float(np.max(spacing[:2])):.2f} mm.",
                spacing_mm=[float(v) for v in spacing],
            )
        )
    if volume.is_resampled:
        findings.append(
            _finding(
                "resampled",
                "info",
                "Dataset was resampled",
                "Dataset was resampled to {:.3f} x {:.3f} x {:.3f} mm to obtain a uniform voxel grid.".format(
                    spacing[0], spacing[1], spacing[2]
                ),
                spacing_mm=[round(float(v), 6) for v in spacing],
            )
        )
    if volume.is_normalized:
        findings.append(
            _finding(
                "normalized",
                "info",
                "Intensities normalised",
                "The volume was intensity-normalised; Hounsfield-unit thresholds are no longer meaningful.",
                intensity_units=volume.intensity_units,
            )
        )
    if mask is not None:
        if not np.any(mask):
            findings.append(
                _finding(
                    "empty_mask",
                    "error",
                    "Empty segmentation",
                    "The segmentation contains no foreground voxels, so no surface can be produced.",
                )
            )
        else:
            voxels = int(np.count_nonzero(mask))
            if voxels < 32:
                findings.append(
                    _finding(
                        "tiny_mask",
                        "warning",
                        "Very small segmentation",
                        f"The mask contains only {voxels} voxel(s). Physical measurements will be "
                        "dominated by discretisation error.",
                        voxels=voxels,
                    )
                )
            touches_border = bool(
                mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
                or mask[:, :, 0].any() or mask[:, :, -1].any()
            )
            if touches_border:
                findings.append(
                    _finding(
                        "mask_touches_border",
                        "info",
                        "Segmentation touches the volume boundary",
                        "The mask reaches the edge of the reconstructed volume, so the surface is open "
                        "there. Enclosed mesh volume is not reported; the voxel volume remains valid.",
                    )
                )
    for message in volume.warnings:
        findings.append(_finding("volume_warning", "warning", "Reconstruction warning", message))
    return findings


def check_segmentation(segmentation: SegmentationResult, volume: Volume) -> list[Finding]:
    findings: list[Finding] = []
    for message in segmentation.warnings:
        findings.append(_finding("segmentation_warning", "warning", "Segmentation warning", message))

    stats = segmentation.params.get("statistics") if isinstance(segmentation.params, dict) else None
    if isinstance(stats, dict) and stats.get("components", 1) > 1:
        findings.append(
            _finding(
                "multiple_components",
                "info",
                "Multiple disconnected components",
                f"The mask contains {stats['components']} disconnected components. Enable 'keep largest N "
                "components' if only one structure is expected.",
                components=stats["components"],
            )
        )
    if segmentation.engine == "classical_fallback":
        findings.append(
            _finding(
                "classical_fallback",
                "warning",
                "Classical fallback used",
                "An AI engine was requested but is unavailable; a classical threshold result is shown "
                "instead. It is not a model prediction.",
            )
        )
    return findings


def summarise(findings: list[Finding]) -> dict[str, Any]:
    counts = {"error": 0, "warning": 0, "info": 0}
    for finding in findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    return {
        "total": len(findings),
        "errors": counts.get("error", 0),
        "warnings": counts.get("warning", 0),
        "info": counts.get("info", 0),
        "status": "error" if counts.get("error") else ("warning" if counts.get("warning") else "ok"),
    }
