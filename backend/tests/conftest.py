"""Shared pytest fixtures.

A single synthetic phantom is generated once per session and reused by the
algorithmic tests.  The API tests drive the real HTTP surface with a
``TestClient``; the store is process-global, so the API tests build their
artefacts through the API rather than sharing objects with the unit tests.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Force every test run into a throwaway data directory.
os.environ["ANATO3D_DATA_DIR"] = tempfile.mkdtemp(prefix="anato3d_test_")

from app.models.domain import Volume  # noqa: E402
from app.services.dicom import reader  # noqa: E402
from app.services.dicom import series as series_service  # noqa: E402
from app.services.reconstruction.volume import build_volume  # noqa: E402
from app.services.synthetic import PhantomOptions, generate_phantom_dataset  # noqa: E402

PHANTOM_OPTIONS = PhantomOptions(
    rows=64,
    columns=64,
    slices=32,
    row_spacing_mm=1.5,
    column_spacing_mm=1.5,
    slice_spacing_mm=2.0,
    orientation="axial",
    include_lesion=True,
    shuffle_filenames=True,
)


@pytest.fixture(scope="session")
def phantom(tmp_path_factory) -> dict:
    """Generate the phantom and expose a DICOM-only dataset directory.

    The generator also writes a ``.npz`` ground-truth mask next to the slices.
    The real upload pipeline only ever stores ``.dcm`` files in the dataset
    directory and registers the reference separately, so the tests mirror that
    by copying the slices into a clean subdirectory.
    """
    root = tmp_path_factory.mktemp("phantom")
    result = generate_phantom_dataset(root, PHANTOM_OPTIONS)

    dicom_only = root / "dicom"
    dicom_only.mkdir()
    for path in result.files:
        shutil.copy2(path, dicom_only / path.name)

    return {"dir": dicom_only, "result": result, "root": root}


@pytest.fixture(scope="session")
def phantom_result(phantom):
    return phantom["result"]


@pytest.fixture(scope="session")
def dataset(phantom):
    scanned = reader.scan_dataset(phantom["dir"])
    return scanned


@pytest.fixture(scope="session")
def analysis(dataset, phantom):
    return series_service.analyze_dataset(dataset, phantom["dir"])


@pytest.fixture(scope="session")
def primary_series(analysis):
    uid = analysis.primary_series_uid
    assert uid is not None
    for entry in analysis.series:
        if entry.info.series_instance_uid == uid:
            return entry
    raise AssertionError(f"primary series {uid!r} was not found in the analysis")


@pytest.fixture(scope="session")
def volume(dataset, phantom, primary_series) -> Volume:
    result = build_volume(dataset, phantom["dir"], primary_series)
    return result.volume


@pytest.fixture(scope="session")
def volume_with_ordering(dataset, phantom, primary_series):
    return build_volume(dataset, phantom["dir"], primary_series)


@pytest.fixture(scope="session")
def ground_truth_mask(phantom_result) -> np.ndarray:
    packed = np.load(phantom_result.ground_truth_path)["mask"]
    shape = tuple(phantom_result.dimensions)
    return np.unpackbits(packed)[: int(np.prod(shape))].astype(bool).reshape(shape)


@pytest.fixture(scope="session")
def preset_params() -> dict:
    from app.services.segmentation.registry import PRESETS

    return {preset["key"]: dict(preset["params"]) for preset in PRESETS}


@pytest.fixture(scope="session")
def segmentation_result(volume, preset_params):
    """Bone segmentation of the phantom, shared by the segmentation/mesh tests."""
    from app.services.segmentation.pipeline import run_segmentation

    return run_segmentation(volume, "threshold", preset_params["ct_bone"])


@pytest.fixture(scope="session")
def mesh_asset(segmentation_result, volume, tmp_path_factory):
    from app.services.mesh.service import generate_mesh

    out = tmp_path_factory.mktemp("mesh")
    return generate_mesh(segmentation_result, volume, output_dir=out)


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)
