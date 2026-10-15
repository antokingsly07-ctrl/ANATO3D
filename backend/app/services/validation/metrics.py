"""Quantitative segmentation metrics.

These are only computed when a ground-truth (reference) mask is supplied.  When
none is available the application reports that explicitly instead of presenting
invented numbers -- see :func:`unavailable_report`.

Implementation notes
--------------------
Distances are computed on the *surface voxels* of each mask using an exact
Euclidean distance transform sampled with the true anisotropic voxel spacing, so
Hausdorff / average surface distances are in millimetres of real anatomy.

* ``Dice`` and ``IoU`` (Jaccard) -- overlap.
* ``Hausdorff distance`` (HD) and ``95th percentile Hausdorff`` (HD95) --
  boundary agreement; HD95 is preferred because it is robust to single outliers.
* ``Average symmetric surface distance`` (ASSD) -- mean boundary error.
* ``Volume difference`` and ``relative volume difference``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from ...core.errors import ValidationError
from ...core.logging_ import get_logger

logger = get_logger("validation.metrics")


@dataclass
class ConfusionCounts:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    def as_dict(self) -> dict[str, int]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
        }


@dataclass
class MetricValue:
    key: str
    label: str
    value: float | None
    unit: str
    interpretation: str
    direction: str = "higher_is_better"  # or "lower_is_better" / "neutral"

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "value": None if self.value is None else round(float(self.value), 6),
            "unit": self.unit,
            "interpretation": self.interpretation,
            "direction": self.direction,
        }


@dataclass
class MetricReport:
    metrics: list[MetricValue] = field(default_factory=list)
    counts: ConfusionCounts | None = None
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": [m.as_dict() for m in self.metrics],
            "counts": self.counts.as_dict() if self.counts else None,
            "notes": list(self.notes),
            "warnings": list(self.warnings),
        }


def _surface(mask: np.ndarray, connectivity: int = 1) -> np.ndarray:
    structure = ndimage.generate_binary_structure(3, connectivity)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return mask & ~eroded


def _directed_distances(source_surface: np.ndarray, target_surface: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    if not np.any(source_surface) or not np.any(target_surface):
        return np.zeros(0, dtype=np.float64)
    distance = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    return distance[source_surface]


def compute_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    spacing_mm: tuple[float, float, float],
    *,
    volume_of_voxel_mm3: float | None = None,
) -> MetricReport:
    """Compare a predicted mask against a reference mask."""
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if prediction.shape != reference.shape:
        raise ValidationError(
            "The ground-truth mask dimensions do not match the segmentation.",
            detail=f"prediction {prediction.shape} vs reference {reference.shape}",
        )
    spacing = np.asarray(spacing_mm, dtype=np.float64)
    report = MetricReport()

    tp = int(np.count_nonzero(prediction & reference))
    fp = int(np.count_nonzero(prediction & ~reference))
    fn = int(np.count_nonzero(~prediction & reference))
    tn = int(prediction.size - tp - fp - fn)
    report.counts = ConfusionCounts(tp, fp, fn, tn)

    if tp + fp + fn == 0:
        report.warnings.append("Both masks are empty; overlap metrics are undefined.")
        return report

    dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    report.metrics.extend(
        [
            MetricValue(
                "dice",
                "Dice similarity coefficient",
                dice,
                "0-1",
                "1.0 means perfect overlap. Above 0.9 is generally considered excellent agreement "
                "for a comparable task.",
            ),
            MetricValue("iou", "Intersection over Union (Jaccard)", iou, "0-1",
                        "1.0 means perfect overlap; more sensitive to small structures than Dice."),
            MetricValue("precision", "Precision", precision, "0-1",
                        "Fraction of predicted voxels that are correct. Low values indicate over-segmentation."),
            MetricValue("recall", "Recall (sensitivity)", recall, "0-1",
                        "Fraction of reference voxels that were found. Low values indicate under-segmentation."),
        ]
    )

    if np.any(prediction) and np.any(reference):
        pred_surface = _surface(prediction)
        ref_surface = _surface(reference)
        d_pred_to_ref = _directed_distances(pred_surface, ref_surface, spacing)
        d_ref_to_pred = _directed_distances(ref_surface, pred_surface, spacing)
        all_distances = np.concatenate([d_pred_to_ref, d_ref_to_pred])

        hd = float(max(d_pred_to_ref.max(initial=0.0), d_ref_to_pred.max(initial=0.0)))
        hd95 = float(
            max(
                np.percentile(d_pred_to_ref, 95) if d_pred_to_ref.size else 0.0,
                np.percentile(d_ref_to_pred, 95) if d_ref_to_pred.size else 0.0,
            )
        )
        assd = float(all_distances.mean()) if all_distances.size else 0.0

        report.metrics.extend(
            [
                MetricValue("hausdorff", "Hausdorff distance (HD)", hd, "mm",
                            "Maximum boundary error. Very sensitive to single outliers.", "lower_is_better"),
                MetricValue("hausdorff95", "95th percentile Hausdorff (HD95)", hd95, "mm",
                            "Boundary error ignoring the worst 5% of surface points; the standard "
                            "robust choice.", "lower_is_better"),
                MetricValue("assd", "Average symmetric surface distance (ASSD)", assd, "mm",
                            "Mean boundary distance in both directions. A practical measure of "
                            "boundary accuracy.", "lower_is_better"),
            ]
        )
        report.notes.append(
            f"Symmetric surface distances computed over {int(all_distances.size)} surface voxels with "
            f"spacing {spacing[0]:.4g} x {spacing[1]:.4g} x {spacing[2]:.4g} mm."
        )
    else:
        report.warnings.append(
            "One of the masks is empty, so boundary distance metrics (HD, HD95, ASSD) could not be computed."
        )

    if volume_of_voxel_mm3 is None:
        volume_of_voxel_mm3 = float(np.prod(spacing))
    pred_volume = np.count_nonzero(prediction) * volume_of_voxel_mm3
    ref_volume = np.count_nonzero(reference) * volume_of_voxel_mm3
    difference = pred_volume - ref_volume
    relative = (difference / ref_volume) if ref_volume > 0 else None

    report.metrics.extend(
        [
            MetricValue("volume_prediction", "Predicted volume", pred_volume / 1000.0, "cm3",
                        "Volume of the segmentation under evaluation.", "neutral"),
            MetricValue("volume_reference", "Reference volume", ref_volume / 1000.0, "cm3",
                        "Volume of the ground-truth mask.", "neutral"),
            MetricValue("volume_difference", "Volume difference", difference / 1000.0, "cm3",
                        "Prediction minus reference. Positive means over-segmentation.", "neutral"),
            MetricValue("relative_volume_difference", "Relative volume difference",
                        None if relative is None else relative * 100.0, "%",
                        "Volume difference as a percentage of the reference volume.", "neutral"),
        ]
    )
    return report


def unavailable_report(reason: str | None = None) -> dict[str, Any]:
    """Explicit, non-fabricated result used when no reference standard exists."""
    return {
        "state": "unavailable",
        "message": "Ground truth unavailable - quantitative segmentation accuracy cannot be calculated.",
        "detail": reason
        or (
            "No reference (ground-truth) mask was supplied for this segmentation. Providing one "
            "(NIfTI or a saved mask archive) enables Dice, IoU, Hausdorff and surface-distance metrics."
        ),
        "metrics": [],
        "counts": None,
        "notes": [],
        "warnings": [],
    }
