"""Marching-cubes mesh generation, refinement and file export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_mesh_has_geometry(mesh_asset):
    mesh = mesh_asset.mesh
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    assert mesh.vertices.shape[1] == 3
    assert mesh.faces.shape[1] == 3
    assert mesh.faces.dtype.kind in ("i", "u")


def test_expected_formats_are_exported(mesh_asset):
    assert {"glb", "stl", "obj"} <= set(mesh_asset.files)
    for path in mesh_asset.files.values():
        file = Path(path)
        assert file.exists()
        assert file.stat().st_size > 0


def test_glb_has_gltf_magic(mesh_asset):
    raw = Path(mesh_asset.files["glb"]).read_bytes()
    assert raw[:4] == b"glTF"


def test_obj_contains_faces(mesh_asset):
    text = Path(mesh_asset.files["obj"]).read_text(encoding="utf-8", errors="ignore")
    assert "f " in text
    assert "v " in text


def test_binary_stl_face_count_matches(mesh_asset):
    raw = Path(mesh_asset.files["stl"]).read_bytes()
    assert len(raw) >= 84
    triangles = int.from_bytes(raw[80:84], "little")
    # trimesh writes binary STL; the header count must match the in-memory mesh.
    assert triangles == mesh_asset.mesh.n_faces


def test_mesh_vertices_are_finite_and_within_the_field_of_view(mesh_asset, volume):
    vertices = np.asarray(mesh_asset.mesh.vertices)
    assert np.all(np.isfinite(vertices))
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    assert np.all(extent > 0)
    # Mesh lives in viewer (mm) space; it can never exceed the imaged extent.
    assert np.all(extent <= max(volume.physical_size_mm) + 1e-3)


def test_voxel_volume_matches_the_mask(mesh_asset, segmentation_result):
    voxel = mesh_asset.stats["measurements"]["voxel"]
    expected_voxels = int(segmentation_result.mask.sum())
    assert voxel["count"] == expected_voxels
    assert voxel["volume_mm3"] == pytest.approx(expected_voxels * 4.5, rel=1e-6)


def test_surface_area_is_positive(mesh_asset):
    area = mesh_asset.stats["measurements"]["mesh"]["surface_area_mm2"]
    assert area is not None and area > 0


def test_watertight_flag_is_reported_honestly(mesh_asset):
    watertight = mesh_asset.stats["measurements"]["mesh"]["is_watertight"]
    assert isinstance(watertight, (bool, np.bool_))
    if not watertight:
        assert any("watertight" in warning.lower() for warning in mesh_asset.warnings)


def test_refinement_report_is_present(mesh_asset):
    refinement = mesh_asset.stats["refinement"]
    assert refinement["output_faces"] == mesh_asset.mesh.n_faces
    assert refinement["input_faces"] >= refinement["output_faces"]


def test_available_formats_match_files(mesh_asset):
    assert mesh_asset.stats["available_formats"] == sorted(mesh_asset.files)


def test_mesh_links_back_to_its_inputs(mesh_asset, segmentation_result, volume):
    assert mesh_asset.segmentation_id == segmentation_result.segmentation_id
    assert mesh_asset.volume_id == volume.volume_id
