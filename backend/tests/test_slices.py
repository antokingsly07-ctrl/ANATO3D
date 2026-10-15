"""Axial / coronal / sagittal plane extraction and anatomical labelling."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.reconstruction.slices import (
    direction_letter,
    direction_name,
    extract_mask_plane,
    extract_plane,
    plane_counts,
)


def test_direction_letters_follow_lps():
    assert direction_letter(np.array([1.0, 0.0, 0.0])) == "L"
    assert direction_letter(np.array([-1.0, 0.0, 0.0])) == "R"
    assert direction_letter(np.array([0.0, 1.0, 0.0])) == "P"
    assert direction_letter(np.array([0.0, -1.0, 0.0])) == "A"
    assert direction_letter(np.array([0.0, 0.0, 1.0])) == "S"
    assert direction_letter(np.array([0.0, 0.0, -1.0])) == "I"


def test_direction_names_are_spelled_out():
    assert direction_name(np.array([0.0, 0.0, 1.0])) == "Superior"
    assert direction_name(np.array([0.0, 0.0, -1.0])) == "Inferior"


def test_plane_counts_match_volume_shape(volume):
    assert plane_counts(volume) == {"axial": 32, "coronal": 64, "sagittal": 64}


def test_axial_plane_shape_and_dtype(volume):
    view = extract_plane(volume, "axial", 16)
    assert (view.rows, view.columns) == (64, 64)
    assert view.array.shape == (64, 64)
    assert view.array.dtype == np.float32
    assert view.count == 32
    assert view.index == 16
    assert view.position_label.startswith("Z =")


def test_coronal_and_sagittal_shapes(volume):
    coronal = extract_plane(volume, "coronal", 32)
    sagittal = extract_plane(volume, "sagittal", 32)
    assert (coronal.rows, coronal.columns) == (32, 64)
    assert (sagittal.rows, sagittal.columns) == (32, 64)
    assert coronal.count == 64
    assert sagittal.count == 64


def test_index_is_clamped_to_range(volume):
    view = extract_plane(volume, "axial", 9999)
    assert view.index == view.count - 1
    view_low = extract_plane(volume, "axial", -50)
    assert view_low.index == 0


def test_axial_labels_are_anatomical(volume):
    labels = extract_plane(volume, "axial", 16).labels
    # Standard radiological axial convention: anterior at the top, patient left on the right.
    assert labels["top"] == "A"
    assert labels["bottom"] == "P"
    assert labels["left"] == "R"
    assert labels["right"] == "L"


def test_coronal_and_sagittal_labels(volume):
    coronal = extract_plane(volume, "coronal", 32).labels
    assert coronal["top"] == "S" and coronal["bottom"] == "I"
    assert coronal["left"] == "R" and coronal["right"] == "L"

    sagittal = extract_plane(volume, "sagittal", 32).labels
    assert sagittal["top"] == "S" and sagittal["bottom"] == "I"
    # Anterior is on the left of a left-sided sagittal view.
    assert sagittal["left"] == "A" and sagittal["right"] == "P"


def test_unknown_plane_is_rejected(volume):
    from app.core.errors import ValidationError

    with pytest.raises(ValidationError):
        extract_plane(volume, "oblique", 0)


def test_mask_plane_matches_image_geometry(volume, segmentation_result):
    image = extract_plane(volume, "coronal", 32)
    mask = extract_mask_plane(segmentation_result.mask, volume, "coronal", 32)
    assert mask.shape == (image.rows, image.columns)
    assert mask.dtype == bool


def test_without_array_omits_payload(volume):
    view = extract_plane(volume, "axial", 5, with_array=False)
    assert view.array is None
    assert view.rows == 64
