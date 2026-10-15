"""Segmentation engines, presets and mask statistics."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.errors import SegmentationError, ValidationError
from app.services.segmentation.pipeline import run_segmentation
from app.services.segmentation.registry import get_engine, list_engines, list_presets


def test_registry_lists_expected_engines(volume):
    engines = {engine["key"]: engine for engine in list_engines(volume)}
    assert {"threshold", "region_growing", "ai", "classical_fallback"} <= set(engines)
    assert engines["threshold"]["availability"]["available"] is True


def test_ai_engine_is_unavailable_without_a_configured_model(volume):
    # No MONAI bundle (and no torch/monai) is configured in the test environment.
    availability = get_engine("ai").availability(volume)
    assert availability.available is False
    assert availability.reason


def test_requesting_ai_falls_back_to_a_labelled_classical_engine(volume, preset_params):
    result = run_segmentation(volume, "ai", preset_params["ct_bone"])
    assert result.engine == "classical_fallback"
    joined = " ".join(result.notes).lower()
    assert "falling back" in joined or "fallback" in joined
    assert "classical" in joined
    assert result.mask.sum() > 0


def test_threshold_selects_only_voxels_inside_the_window(volume):
    params = {
        "mode": "absolute",
        "threshold_min": 200.0,
        "threshold_max": 3071.0,
        "fill_holes": False,
        "close_radius": 0,
        "open_radius": 0,
        "min_component_mm3": 0.0,
    }
    result = run_segmentation(volume, "threshold", params)
    selected = np.asarray(volume.array)[result.mask]
    assert selected.size > 0
    assert float(selected.min()) >= 200.0
    assert float(selected.max()) <= 3071.0


def test_threshold_matches_manual_computation(volume):
    params = {
        "mode": "absolute",
        "threshold_min": 200.0,
        "threshold_max": 3071.0,
        "fill_holes": False,
        "close_radius": 0,
        "open_radius": 0,
        "min_component_mm3": 0.0,
    }
    result = run_segmentation(volume, "threshold", params)
    expected = (np.asarray(volume.array) >= 200.0) & (np.asarray(volume.array) <= 3071.0)
    assert np.array_equal(result.mask, expected)


def test_empty_threshold_raises(volume):
    with pytest.raises(SegmentationError):
        run_segmentation(volume, "threshold", {"mode": "absolute", "threshold_min": 9000.0, "threshold_max": 9500.0})


def test_percentile_mode_selects_high_intensity(volume):
    params = {
        "mode": "percentile",
        "lower_percentile": 90.0,
        "upper_percentile": 100.0,
        "fill_holes": False,
        "close_radius": 0,
        "open_radius": 0,
        "min_component_mm3": 0.0,
    }
    result = run_segmentation(volume, "threshold", params)
    assert result.mask.sum() > 0
    assert "percentile" in " ".join(result.notes).lower()


def test_unknown_engine_is_rejected(volume):
    with pytest.raises(ValidationError):
        run_segmentation(volume, "not_a_real_engine", {})


def test_region_growing_produces_a_connected_mask(volume, preset_params):
    result = run_segmentation(volume, "region_growing", preset_params["ct_region_grow_bone"])
    assert result.engine == "region_growing"
    assert result.mask.sum() > 0
    assert result.params["statistics"]["components"] >= 1


def test_min_component_filter_never_grows_the_mask(volume):
    base_params = {
        "mode": "absolute",
        "threshold_min": 200.0,
        "threshold_max": 3071.0,
        "fill_holes": False,
        "close_radius": 0,
        "open_radius": 0,
        "min_component_mm3": 0.0,
    }
    base = run_segmentation(volume, "threshold", base_params)
    filtered = run_segmentation(
        volume, "threshold", {**base_params, "min_component_mm3": 1000.0}
    )
    assert filtered.mask.sum() <= base.mask.sum()
    assert (
        filtered.params["statistics"]["components"]
        <= base.params["statistics"]["components"]
    )


def test_bone_preset_result_is_consistent(volume, segmentation_result):
    assert segmentation_result.volume_id == volume.volume_id
    assert segmentation_result.label == "Bone"
    assert segmentation_result.engine == "threshold"
    assert segmentation_result.mask.dtype == bool
    assert segmentation_result.mask.shape == volume.shape


def test_mask_statistics_are_self_consistent(segmentation_result):
    stats = segmentation_result.params["statistics"]
    voxels = int(segmentation_result.mask.sum())
    assert stats["voxel_count"] == voxels
    assert stats["volume_mm3"] == pytest.approx(voxels * 4.5, rel=1e-6)
    assert stats["volume_cm3"] == pytest.approx(voxels * 4.5 / 1000.0, rel=1e-6)


def test_presets_are_annotated_for_modality():
    presets = list_presets("CT")
    by_key = {preset["key"]: preset for preset in presets}
    assert "ct_bone" in by_key
    assert by_key["ct_bone"]["applicable"] is True

    mr_presets = list_presets("MR")
    by_key_mr = {preset["key"]: preset for preset in mr_presets}
    assert by_key_mr["ct_bone"]["applicable"] is False
    assert "applicability_note" in by_key_mr["ct_bone"]
