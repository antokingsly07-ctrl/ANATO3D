"""Volume reconstruction: geometry, intensity and resampling.

These tests pin the conventions the whole application depends on:

* array shape is ``(k, j, i) = (slice, row, column)``,
* ``physical = origin + direction @ (index * spacing)``,
* the direction matrix is the DICOM/ITK LPS frame (determinant ``-1``).
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.reconstruction.volume import build_volume, resample_volume


def test_shape_and_spacing(volume):
    assert volume.shape == (32, 64, 64)
    assert volume.spacing == pytest.approx((2.0, 1.5, 1.5))


def test_direction_is_dicom_lps_frame(volume):
    # Columns are [slice, row, column] direction cosines in LPS; a faithful
    # DICOM frame has determinant -1 (not the right-handed +1).
    assert np.linalg.det(volume.direction) == pytest.approx(-1.0, abs=1e-9)


def test_direction_columns_are_unit_and_orthogonal(volume):
    direction = volume.direction
    gram = direction.T @ direction
    assert gram == pytest.approx(np.eye(3), abs=1e-9)


def test_voxel_volume_and_physical_size(volume):
    assert volume.voxel_volume_mm3 == pytest.approx(2.0 * 1.5 * 1.5)
    assert volume.physical_size_mm == pytest.approx((64.0, 96.0, 96.0))


def test_orientation_label(volume):
    assert "axial" in volume.orientation_label.lower()


def test_intensity_is_hounsfield(volume):
    assert volume.modality == "CT"
    assert volume.intensity_units == "HU"
    array = np.asarray(volume.array)
    # The phantom spans air/lung (very negative) to cortical bone (very positive).
    assert float(array.min()) < -500.0
    assert float(array.max()) > 500.0


def test_index_to_lps_matches_origin(volume):
    point = volume.index_to_lps(np.zeros(3))
    assert point == pytest.approx(volume.origin, abs=1e-9)


def test_index_to_lps_advances_along_slice_axis_by_spacing(volume):
    a = volume.index_to_lps(np.array([0.0, 0.0, 0.0]))
    b = volume.index_to_lps(np.array([1.0, 0.0, 0.0]))
    assert np.linalg.norm(b - a) == pytest.approx(volume.spacing[0], abs=1e-6)


def test_dtype_is_memory_appropriate(volume):
    assert volume.array.dtype in (np.dtype("float32"), np.dtype("int16"))


def test_ordering_positions_are_strictly_increasing(volume_with_ordering):
    positions = np.asarray(volume_with_ordering.ordering.slice_positions_mm)
    assert np.all(np.diff(positions) > 0)


def test_reconstruction_report_records_source_and_target(volume_with_ordering):
    report = volume_with_ordering.report
    assert report["slice_count"] == 32
    assert report["reconstructed_slice_count"] == 32
    assert report["dimensions_order"] == "slices x rows x columns"
    assert report["voxel_spacing_order"] == "slice x row x column"


def test_resample_to_isotropic_changes_spacing_and_grows(volume):
    resampled, spacing = resample_volume(volume.array, volume.spacing, (1.0, 1.0, 1.0))
    assert spacing == pytest.approx((1.0, 1.0, 1.0))
    assert resampled.shape[0] > volume.shape[0]
    assert resampled.shape[1] > volume.shape[1]


def test_resample_is_a_noop_at_native_spacing(volume):
    resampled, spacing = resample_volume(volume.array, volume.spacing, volume.spacing)
    assert resampled.shape == volume.shape
    assert resampled.dtype == np.asarray(volume.array).dtype
    assert spacing == pytest.approx(volume.spacing)
