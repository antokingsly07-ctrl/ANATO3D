"""Slice ordering must be derived from patient-space geometry only.

The synthetic phantom shuffles both filenames and (in the corrupting tests)
InstanceNumber, so a passing suite proves the ordering code cannot be relying
on either of them.
"""

from __future__ import annotations

import copy
import random

import pytest

from app.core.errors import IncompleteOrientationError
from app.services.dicom.series_ordering import order_series


def _geometries(primary_series):
    """Deep copies so tests can corrupt values without touching the fixtures."""
    return [
        copy.deepcopy(entry.geometry)
        for entry in primary_series.files
        if entry.geometry is not None
    ]


def test_projection_is_monotonic(primary_series):
    ordering = order_series(_geometries(primary_series))
    keys = [geometry.sort_key for geometry in ordering.ordered]
    assert keys == sorted(keys), "slices must be ordered by their patient-space projection"
    assert len(ordering.ordered) == len(primary_series.files)


def test_order_is_independent_of_input_order(primary_series):
    geometries = _geometries(primary_series)
    baseline = [g.sop_instance_uid for g in order_series(geometries).ordered]

    shuffled = list(geometries)
    random.Random(1234).shuffle(shuffled)
    after = [g.sop_instance_uid for g in order_series(shuffled).ordered]

    assert baseline == after, "ordering must not depend on the order files are discovered"


def test_order_is_independent_of_instance_number(primary_series):
    """Deliberately corrupt InstanceNumber; patient positions must win."""
    geometries = _geometries(primary_series)
    baseline = [g.sop_instance_uid for g in order_series(geometries).ordered]

    for index, geometry in enumerate(geometries):
        # Assign instance numbers in the reverse of the true physical order.
        geometry.instance_number = len(geometries) - index

    after = [g.sop_instance_uid for g in order_series(geometries).ordered]
    assert baseline == after


def test_uniform_spacing_is_reported(primary_series):
    ordering = order_series(_geometries(primary_series))
    assert ordering.irregular_spacing is False
    assert ordering.slice_spacing_mm == pytest.approx(2.0, abs=1e-6)
    assert ordering.missing_slices == 0
    assert ordering.gaps_detected == 0
    assert ordering.ordering_method == "spatial"


def test_orientation_is_axial(primary_series):
    ordering = order_series(_geometries(primary_series))
    label = ordering.orientation_label.lower()
    assert "axial" in label or "transverse" in label
    # The phantom is a clean axis-aligned series, so it must not be oblique.
    assert "oblique" not in label


def test_partial_missing_geometry_is_excluded_and_warned(primary_series):
    geometries = _geometries(primary_series)
    geometries[0].image_position_patient = None

    ordering = order_series(geometries)

    assert len(ordering.ordered) == len(geometries) - 1
    assert any("excluded" in warning for warning in ordering.warnings)


def test_all_missing_geometry_raises_without_fallback(primary_series):
    geometries = _geometries(primary_series)
    for geometry in geometries:
        geometry.image_position_patient = None

    with pytest.raises(IncompleteOrientationError) as excinfo:
        order_series(geometries)
    assert "fallback" in (excinfo.value.detail or "").lower()


def test_instance_number_fallback_is_opt_in_and_flagged(primary_series):
    geometries = _geometries(primary_series)
    for geometry in geometries:
        geometry.image_position_patient = None

    ordering = order_series(geometries, allow_instance_number_fallback=True)

    assert ordering.ordering_method == "instance_number_fallback"
    assert any("InstanceNumber" in warning or "fallback" in warning.lower() for warning in ordering.warnings)
