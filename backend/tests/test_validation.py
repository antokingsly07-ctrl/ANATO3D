"""Reconstruction/segmentation validation and quantitative metrics.

The crucial contract: metrics are only ever produced from a real reference
mask.  Without one the report must say so explicitly rather than invent numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.errors import ValidationError
from app.services.validation.service import run_validation


def _run(dataset, analysis, series, ordering, volume, segmentation, **extra):
    return run_validation(
        dataset=dataset,
        analysis=analysis,
        series=series,
        ordering=ordering,
        volume=volume,
        segmentation=segmentation,
        **extra,
    )


def test_validation_without_reference_reports_unavailable(
    dataset, analysis, primary_series, volume_with_ordering, segmentation_result
):
    report = _run(
        dataset,
        analysis,
        primary_series.info,
        volume_with_ordering.ordering,
        volume_with_ordering.volume,
        segmentation_result,
    )
    ground_truth = report["ground_truth"]
    assert ground_truth["state"] == "unavailable"
    assert "Ground truth unavailable" in ground_truth["message"]
    assert "cannot be calculated" in ground_truth["message"]
    assert ground_truth["metrics"] == []


def test_validation_runs_every_section_with_severity_counts(
    dataset, analysis, primary_series, volume_with_ordering, segmentation_result
):
    report = _run(
        dataset,
        analysis,
        primary_series.info,
        volume_with_ordering.ordering,
        volume_with_ordering.volume,
        segmentation_result,
    )
    assert set(report["sections_run"]) == {
        "dataset",
        "series",
        "ordering",
        "volume",
        "segmentation",
    }
    summary = report["summary"]
    assert summary["status"] in ("ok", "warning", "error")
    assert summary["total"] == len(report["findings"])
    for finding in report["findings"]:
        assert finding["severity"] in ("error", "warning", "info")
        assert finding["code"] and finding["title"]
    assert report["disclaimer"]


def test_metrics_against_reference_are_real_and_bounded(
    dataset, analysis, primary_series, volume_with_ordering, segmentation_result, ground_truth_mask
):
    report = _run(
        dataset,
        analysis,
        primary_series.info,
        volume_with_ordering.ordering,
        volume_with_ordering.volume,
        segmentation_result,
        reference_mask=ground_truth_mask,
        reference_source="synthetic phantom",
    )
    ground_truth = report["ground_truth"]
    assert ground_truth["state"] == "computed"
    metrics = {metric["key"]: metric["value"] for metric in ground_truth["metrics"]}

    assert 0.5 < metrics["dice"] <= 1.0
    assert 0.0 < metrics["iou"] <= 1.0
    assert metrics["hausdorff95"] >= 0.0
    assert metrics["hausdorff"] >= metrics["hausdorff95"] - 1e-6
    assert ground_truth["counts"]["true_positive"] > 0
    assert ground_truth["reference_source"] == "synthetic phantom"


def test_identical_masks_give_perfect_overlap(
    dataset, analysis, primary_series, volume_with_ordering, segmentation_result
):
    report = _run(
        dataset,
        analysis,
        primary_series.info,
        volume_with_ordering.ordering,
        volume_with_ordering.volume,
        segmentation_result,
        reference_mask=segmentation_result.mask,
        reference_source="same mask",
    )
    metrics = {metric["key"]: metric["value"] for metric in report["ground_truth"]["metrics"]}
    assert metrics["dice"] == pytest.approx(1.0)
    assert metrics["iou"] == pytest.approx(1.0)
    assert metrics["hausdorff95"] == pytest.approx(0.0)


def test_metrics_require_a_volume_and_segmentation(dataset):
    with pytest.raises(ValidationError):
        run_validation(
            dataset=dataset,
            analysis=None,
            series=None,
            ordering=None,
            volume=None,
            segmentation=None,
            reference_mask=np.zeros((4, 4, 4), dtype=bool),
        )
