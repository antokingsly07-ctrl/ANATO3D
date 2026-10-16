"""NIfTI -> DICOM conversion (used when a user uploads a volume instead of a series)."""

from __future__ import annotations

import numpy as np
import pydicom
import pytest

from app.services.dicom import nifti, reader

SPACING = (1.5, 1.5, 2.0)
ORIGIN = np.array([-100.0, -50.0, 30.0])


@pytest.fixture()
def volume(tmp_path):
    """An 8x10x12 HU-like volume saved as ``.nii.gz`` with a known affine."""
    import nibabel as nib

    data = np.random.default_rng(7).uniform(-1000.0, 3000.0, size=(8, 10, 12)).astype(np.float32)
    affine = np.diag([SPACING[0], SPACING[1], SPACING[2], 1.0])
    affine[:3, 3] = ORIGIN
    path = tmp_path / "volume.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path, data


def test_is_nifti_matches_suffixes(tmp_path):
    assert nifti.is_nifti(tmp_path / "a.nii.gz")
    assert nifti.is_nifti(tmp_path / "a.nii")
    assert not nifti.is_nifti(tmp_path / "a.dcm")
    assert not nifti.is_nifti(tmp_path / "a.nii.txt")


def test_conversion_preserves_geometry_and_values(volume, tmp_path):
    path, data = volume
    out = tmp_path / "out"
    files = nifti.convert_nifti_series(path, out)

    assert len(files) == data.shape[2]

    first = pydicom.dcmread(str(files[0]))
    assert first.Rows == data.shape[0]
    assert first.Columns == data.shape[1]
    assert first.Modality == "CT"
    assert first.SOPClassUID.startswith("1.2.840.10008.5.1.4.1.1.2")
    np.testing.assert_allclose(first.PixelSpacing, [SPACING[0], SPACING[1]], atol=1e-6)
    np.testing.assert_allclose(first.ImageOrientationPatient, [1, 0, 0, 0, 1, 0], atol=1e-6)

    # Intensities survive the stored-as-unsigned + rescale round trip.
    physical = first.pixel_array.astype(np.float64) * first.RescaleSlope + first.RescaleIntercept
    np.testing.assert_allclose(physical, data[:, :, 0], atol=1.0)

    # Slice k sits at origin + k * slice_spacing along the third axis.
    last = pydicom.dcmread(str(files[-1]))
    expected = ORIGIN + (data.shape[2] - 1) * SPACING[2] * np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(last.ImagePositionPatient, expected, atol=1e-6)


def test_converted_series_is_scannable(volume, tmp_path):
    path, data = volume
    out = tmp_path / "out"
    nifti.convert_nifti_series(path, out)
    scanned = reader.scan_dataset(out)
    assert len(scanned.dicom_files) == data.shape[2]
    assert scanned.file_count == data.shape[2]


def test_wide_value_range_uses_slope_instead_of_clipping(tmp_path):
    import nibabel as nib

    data = np.zeros((4, 4, 4), dtype=np.float32)
    data[0, 0, 0] = 0.0
    data[3, 3, 3] = 100000.0
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    path = tmp_path / "wide.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(path))

    files = nifti.convert_nifti_series(path, tmp_path / "wide_out")
    slice_with_max = pydicom.dcmread(str(files[-1]))
    value = (
        slice_with_max.pixel_array.astype(np.float64) * slice_with_max.RescaleSlope
        + slice_with_max.RescaleIntercept
    )
    assert value[3, 3] == pytest.approx(100000.0, rel=1e-4)
