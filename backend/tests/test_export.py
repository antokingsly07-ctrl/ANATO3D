"""Export of volumes, masks, meshes, reports and the combined bundle."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from app.services.export.service import (
    build_bundle,
    export_json,
    export_mask_nifti,
    export_mask_npz,
    export_mesh_asset,
    export_volume_nifti,
    export_volume_npz,
    volume_affine,
)


def test_volume_affine_maps_to_ras(volume):
    affine = volume_affine(volume)
    assert affine.shape == (4, 4)
    lps_to_ras = np.diag([-1.0, -1.0, 1.0])
    assert affine[:3, 3] == pytest.approx(lps_to_ras @ volume.origin, abs=1e-9)
    # the three column vectors recover the true voxel spacing
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    assert sorted(spacing) == pytest.approx(sorted(volume.spacing), abs=1e-6)


def test_nifti_volume_roundtrip(volume, tmp_path):
    nib = pytest.importorskip("nibabel")
    path = export_volume_nifti(volume, tmp_path / "volume.nii.gz")
    image = nib.load(str(path))
    # stored in (i, j, k) = (column, row, slice) order, so the shape is reversed
    assert tuple(image.shape) == tuple(reversed(volume.shape))
    assert image.header.get_xyzt_units()[0] == "mm"


def test_npz_volume_roundtrip(volume, tmp_path):
    path = export_volume_npz(volume, tmp_path / "volume.npz")
    data = np.load(path)
    assert data["array"].shape == volume.shape
    assert data["spacing"] == pytest.approx(volume.spacing)
    assert data["origin"] == pytest.approx(volume.origin)


def test_mask_npz_roundtrip(segmentation_result, tmp_path):
    path = export_mask_npz(segmentation_result.mask, tmp_path / "mask.npz")
    data = np.load(path)
    shape = tuple(int(v) for v in data["shape"])
    assert shape == segmentation_result.mask.shape
    restored = (
        np.unpackbits(data["mask"])[: int(np.prod(shape))].astype(bool).reshape(shape)
    )
    assert np.array_equal(restored, segmentation_result.mask)


def test_nifti_mask_roundtrip(volume, segmentation_result, tmp_path):
    nib = pytest.importorskip("nibabel")
    path = export_mask_nifti(segmentation_result.mask, volume, tmp_path / "mask.nii.gz")
    image = nib.load(str(path))
    assert tuple(image.shape) == tuple(reversed(volume.shape))
    values = np.unique(np.asarray(image.dataobj))
    assert set(values.tolist()) <= {0, 1}


def test_json_export_serialises_numpy(tmp_path):
    path = export_json({"a": np.float32(1.5), "b": np.arange(3)}, tmp_path / "meta.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["a"] == pytest.approx(1.5)
    assert payload["b"] == [0, 1, 2]


def test_mesh_asset_exports_glb(mesh_asset, tmp_path):
    path = export_mesh_asset(mesh_asset, tmp_path / "mesh.glb", "glb")
    assert path.read_bytes()[:4] == b"glTF"


def test_bundle_contains_every_requested_artefact(
    volume, segmentation_result, mesh_asset, tmp_path
):
    path = build_bundle(
        tmp_path / "bundle.zip",
        volume=volume,
        segmentation=segmentation_result,
        mesh=mesh_asset,
        reports={"validation": {"status": "ok"}},
    )

    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        assert {
            "volume/reconstructed_volume.npy",
            "volume/reconstruction_metadata.json",
            "segmentation/mask.npz",
            "segmentation/segmentation_metadata.json",
            "reports/validation.json",
            "manifest.json",
            "README.txt",
        } <= names
        assert any(name.startswith("mesh/structure.") for name in names)

        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["application"] == "ANATO3D"
        assert "volume/reconstructed_volume.npy" in manifest["contents"]

        stored = np.load(io.BytesIO(archive.read("volume/reconstructed_volume.npy")))
        assert stored.shape == tuple(reversed(volume.shape))
