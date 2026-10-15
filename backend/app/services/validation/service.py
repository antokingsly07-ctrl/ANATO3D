"""Validation orchestration.

Combines the automatic consistency checks with optional ground-truth metrics and
produces the report structure the UI and the exporters consume.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ...core.errors import ValidationError
from ...core.logging_ import get_logger
from ...core.storage import new_id
from ...models import DatasetRef, SegmentationResult, SeriesInfo, Volume
from ..dicom.series import DatasetAnalysis
from ..dicom.series_ordering import OrderingResult
from . import checks as checks_module
from . import metrics as metrics_module
from .ground_truth import GroundTruth, align_to_shape, load_ground_truth

logger = get_logger("validation.service")


def run_validation(
    *,
    dataset: DatasetRef | None,
    analysis: DatasetAnalysis | None,
    series: SeriesInfo | None,
    ordering: OrderingResult | None,
    volume: Volume | None,
    segmentation: SegmentationResult | None,
    reference_mask: np.ndarray | None = None,
    reference_source: str | None = None,
    sections: tuple[str, ...] = ("dataset", "series", "ordering", "volume", "segmentation"),
    progress=None,
) -> dict[str, Any]:
    started = time.time()
    findings: list[checks_module.Finding] = []

    if progress:
        progress.step("validation", 0.80, "Running reconstruction and segmentation consistency checks")

    if "dataset" in sections and dataset is not None:
        findings.extend(checks_module.check_dataset(dataset, analysis))
    if "series" in sections and series is not None:
        findings.extend(checks_module.check_series(series))
    if "ordering" in sections and ordering is not None and series is not None:
        findings.extend(checks_module.check_ordering(ordering, series))
    if "volume" in sections and volume is not None:
        findings.extend(checks_module.check_volume(volume, mask=segmentation.mask if segmentation else None))
    if "segmentation" in sections and segmentation is not None and volume is not None:
        findings.extend(checks_module.check_segmentation(segmentation, volume))

    summary = checks_module.summarise(findings)

    ground_truth_block: dict[str, Any]
    if reference_mask is not None:
        if volume is None or segmentation is None:
            raise ValidationError("A volume and a segmentation are required to compute metrics.")
        report = metrics_module.compute_metrics(
            segmentation.mask,
            reference_mask,
            spacing_mm=tuple(float(v) for v in volume.spacing),  # type: ignore[arg-type]
            volume_of_voxel_mm3=float(volume.voxel_volume_mm3),
        )
        ground_truth_block = {
            "state": "computed",
            "reference_source": reference_source or "uploaded",
            "message": "Metrics computed against the supplied reference mask.",
            **report.as_dict(),
        }
        if progress:
            dice = next((m.value for m in report.metrics if m.key == "dice"), None)
            progress.step(
                "validation",
                0.88,
                "Ground-truth metrics computed"
                + (f" (Dice = {dice:.4f})" if dice is not None else ""),
            )
    else:
        available = [
            label
            for label, present in (
                ("dataset", dataset is not None),
                ("volume", volume is not None),
                ("segmentation", segmentation is not None),
            )
            if present
        ]
        ground_truth_block = metrics_module.unavailable_report(
            "No reference mask was supplied. "
            + (f"Validation context available: {', '.join(available)}." if available else "")
        )

    report = {
        "validation_id": new_id("val"),
        "created_at": time.time(),
        "duration_s": round(time.time() - started, 3),
        "sections_run": list(sections),
        "summary": summary,
        "findings": [f.as_dict() for f in findings],
        "ground_truth": ground_truth_block,
        "dataset_information": _dataset_information(dataset, analysis, series, volume, segmentation),
        "interpretation_notes": _interpretation_notes(summary, ground_truth_block),
        "disclaimer": (
            "This application is a research prototype and is NOT a certified medical device. "
            "It has not been cleared or approved by any regulator and must not be used for "
            "diagnosis or treatment decisions."
        ),
    }
    return report


def _dataset_information(
    dataset: DatasetRef | None,
    analysis: DatasetAnalysis | None,
    series: SeriesInfo | None,
    volume: Volume | None,
    segmentation: SegmentationResult | None,
) -> dict[str, Any]:
    info: dict[str, Any] = {}
    if dataset is not None:
        info["files_scanned"] = dataset.file_count
        info["dicom_files"] = len(dataset.dicom_files)
    if analysis is not None:
        info["studies"] = len(analysis.studies)
        info["series"] = len(analysis.series)
    if series is not None:
        info["modality"] = series.modality
        info["slice_count"] = series.slice_count
        info["orientation"] = series.orientation_label
        info["slice_spacing_mm"] = series.slice_spacing
        info["rows"] = series.rows
        info["columns"] = series.columns
    if volume is not None:
        info["volume_dimensions"] = list(volume.shape)
        info["voxel_spacing_mm"] = [round(float(v), 6) for v in volume.spacing]
        info["voxel_volume_mm3"] = round(float(volume.voxel_volume_mm3), 8)
        info["resampled"] = bool(volume.is_resampled)
        info["intensity_units"] = volume.intensity_units
    if segmentation is not None:
        info["segmentation_engine"] = segmentation.engine
        stats = segmentation.params.get("statistics") if isinstance(segmentation.params, dict) else None
        if isinstance(stats, dict):
            info["segmentation_voxels"] = stats.get("voxel_count")
            info["segmentation_volume_cm3"] = stats.get("volume_cm3")
    return info


def _interpretation_notes(summary: dict[str, Any], ground_truth: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    if summary["errors"]:
        notes.append(
            f"{summary['errors']} error-level finding(s) require attention before results can be "
            "considered reliable."
        )
    if summary["warnings"]:
        notes.append(
            f"{summary['warnings']} warning(s) indicate conditions that may reduce accuracy."
        )
    if summary["status"] == "ok":
        notes.append("No consistency problems were detected in the reconstructed data.")

    if ground_truth.get("state") == "unavailable":
        notes.append(
            "Quantitative accuracy metrics are unavailable because no ground-truth segmentation was "
            "provided. No accuracy values are estimated or inferred."
        )
    elif ground_truth.get("state") == "computed":
        dice = next(
            (m["value"] for m in ground_truth.get("metrics", []) if m.get("key") == "dice"),
            None,
        )
        if dice is not None:
            if dice >= 0.9:
                verdict = "excellent agreement with the reference"
            elif dice >= 0.8:
                verdict = "good agreement with the reference"
            elif dice >= 0.7:
                verdict = "moderate agreement with the reference"
            else:
                verdict = "limited agreement with the reference"
            notes.append(f"Dice = {dice:.4f} indicates {verdict}.")
        notes.append(
            "These metrics describe agreement with the supplied reference mask. They do not "
            "establish clinical accuracy."
        )
    return notes


def load_reference_from_path(path: Path, volume: Volume) -> GroundTruth:
    ground_truth = load_ground_truth(path)
    aligned = align_to_shape(ground_truth, tuple(volume.shape))  # type: ignore[arg-type]
    if not np.any(aligned.mask):
        raise ValidationError(
            "The supplied ground-truth mask contains no foreground voxels after alignment.",
            detail=f"Source: {aligned.source}",
        )
    return aligned
