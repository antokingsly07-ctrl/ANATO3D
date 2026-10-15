"""End-to-end HTTP tests driving the real JSON API.

These exercise the same path a browser takes, including the asynchronous
``?wait=false`` job flow.  The synthetic phantom is used so the whole pipeline
can run deterministically without any external data.
"""

from __future__ import annotations

import base64
import json
import time

import pytest

SYNTHETIC = {
    "rows": 64,
    "columns": 64,
    "slices": 32,
    "row_spacing_mm": 1.5,
    "column_spacing_mm": 1.5,
    "slice_spacing_mm": 2.0,
    "orientation": "axial",
}


def _completed(response, label: str) -> dict:
    assert response.status_code == 200, f"{label}: HTTP {response.status_code}: {response.text}"
    payload = response.json()
    assert payload["status"] == "completed", f"{label}: {payload.get('status')} {payload.get('error')}"
    assert isinstance(payload["result"], dict)
    return payload["result"]


def _synthetic(client, **overrides) -> dict:
    response = client.post("/api/dicom/synthetic", json={**SYNTHETIC, **overrides})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["label"] == "SYNTHETIC"
    return body


def _analyze(client, dataset_id: str) -> dict:
    return _completed(
        client.post("/api/dicom/analyze", json={"dataset_id": dataset_id}, params={"wait": True}),
        "analyze",
    )


def _reconstruct(client, dataset_id: str, series_uid: str) -> dict:
    return _completed(
        client.post(
            "/api/reconstruction/create",
            json={"dataset_id": dataset_id, "series_instance_uid": series_uid},
            params={"wait": True},
        ),
        "reconstruct",
    )


# ---------------------------------------------------------------------------


def test_health_reports_capabilities(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["ai_segmentation_available"] is False
    assert body["ai_unavailable_reason"]
    assert "max_upload_bytes" in body["limits"]
    assert "data_dir" in body["storage"]


def test_synthetic_dataset_is_flagged_as_synthetic(client):
    body = _synthetic(client)
    assert body["slice_count"] == 32
    assert body["dimensions"] == [32, 64, 64]
    assert body["ground_truth_available"] is True
    assert body["ground_truth_id"]
    assert any("synthetic" in note.lower() for note in body["notes"])


def test_analysis_orders_slices_by_patient_position(client):
    dataset_id = _synthetic(client)["dataset_id"]
    result = _analyze(client, dataset_id)
    projections = [entry["projection_mm"] for entry in result["ordering_preview"]]
    assert len(projections) == 32
    assert projections == sorted(projections)
    assert result["primary_series_uid"]


def test_full_workflow(client):
    synthetic = _synthetic(client)
    dataset_id = synthetic["dataset_id"]
    ground_truth_id = synthetic["ground_truth_id"]

    analysis = _analyze(client, dataset_id)
    series_uid = analysis["primary_series_uid"]

    recon = _reconstruct(client, dataset_id, series_uid)
    volume_id = recon["volume_id"]
    assert recon["geometry"]["dimensions"] == [32, 64, 64]
    assert recon["geometry"]["spacing_mm"][0] == pytest.approx(2.0)
    assert len(recon["slice_preview"]) == 3

    # interactive slice + PNG thumbnail
    slice_body = client.get(
        f"/api/reconstruction/{volume_id}/slice", params={"plane": "axial", "index": 16}
    ).json()
    raw = base64.b64decode(slice_body["data"])
    itemsize = 2 if slice_body["dtype"] == "int16" else 4
    assert len(raw) == slice_body["rows"] * slice_body["columns"] * itemsize
    assert slice_body["position_label"]

    thumbnail = client.get(
        f"/api/reconstruction/{volume_id}/thumbnail", params={"plane": "sagittal"}
    )
    assert thumbnail.content[:8] == b"\x89PNG\r\n\x1a\n"

    # segmentation + mask slice
    segmentation = _completed(
        client.post(
            "/api/segmentation/run",
            json={"volume_id": volume_id, "engine": "threshold", "preset": "ct_bone"},
            params={"wait": True},
        ),
        "segment",
    )
    segmentation_id = segmentation["segmentation_id"]
    assert segmentation["statistics"]["voxel_count"] > 0

    mask_slice = client.get(
        f"/api/segmentation/{segmentation_id}/mask", params={"plane": "axial", "index": 16}
    ).json()
    assert mask_slice["encoding"] == "packbits-base64"

    # mesh + GLB download
    mesh = _completed(
        client.post("/api/mesh/generate", json={"segmentation_id": segmentation_id}, params={"wait": True}),
        "mesh",
    )
    mesh_id = mesh["mesh_id"]
    assert mesh["face_count"] > 0
    assert "glb" in mesh["available_formats"]
    glb = client.get(f"/api/mesh/{mesh_id}/glb")
    assert glb.content[:4] == b"glTF"

    # analysis
    analysis_payload = client.get(f"/api/analysis/{mesh_id}")
    assert analysis_payload.status_code == 200
    assert analysis_payload.json()["voxel"]["volume_mm3"] > 0

    # validation against the known synthetic ground truth
    validation = _completed(
        client.post(
            "/api/validation/run",
            json={
                "dataset_id": dataset_id,
                "volume_id": volume_id,
                "segmentation_id": segmentation_id,
                "ground_truth_id": ground_truth_id,
            },
            params={"wait": True},
        ),
        "validation",
    )
    assert validation["ground_truth"]["state"] == "computed"
    metric_keys = {metric["key"] for metric in validation["ground_truth"]["metrics"]}
    assert {"dice", "iou", "hausdorff95"} <= metric_keys

    # export manifest + bundle
    manifest = client.get(
        f"/api/export/{dataset_id}/manifest",
        params={"volume_id": volume_id, "segmentation_id": segmentation_id, "mesh_id": mesh_id},
    )
    assert manifest.status_code == 200
    items = {item["key"]: item for item in manifest.json()["items"]}
    assert items["bundle"]["available"]

    bundle = client.get(
        f"/api/export/{dataset_id}/file/bundle",
        params={"volume_id": volume_id, "segmentation_id": segmentation_id, "mesh_id": mesh_id},
    )
    assert bundle.status_code == 200
    assert bundle.content[:2] == b"PK"


def test_reconstruction_runs_asynchronously_and_is_pollable(client):
    dataset_id = _synthetic(client)["dataset_id"]
    series_uid = _analyze(client, dataset_id)["primary_series_uid"]

    start = client.post(
        "/api/reconstruction/create",
        json={"dataset_id": dataset_id, "series_instance_uid": series_uid},
        params={"wait": False},
    )
    assert start.status_code == 200
    envelope = start.json()
    job_id = envelope["job_id"]
    assert envelope["status"] in ("pending", "running", "completed")

    final = None
    for _ in range(400):
        final = client.get(f"/api/jobs/{job_id}").json()
        if final["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)

    assert final is not None
    assert final["status"] == "completed", final.get("error")
    assert final["result"]["volume_id"]

    jobs = client.get("/api/jobs").json()
    assert any(job["job_id"] == job_id for job in jobs)


def test_unknown_dataset_returns_404(client):
    response = client.post("/api/dicom/analyze", json={"dataset_id": "ds_does_not_exist"}, params={"wait": True})
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


def test_unknown_volume_slice_returns_404(client):
    response = client.get(
        "/api/reconstruction/vol_missing/slice", params={"plane": "axial", "index": 0}
    )
    assert response.status_code == 404


def test_malformed_payload_returns_422(client):
    response = client.post("/api/dicom/analyze", json={})
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_errors_never_leak_internals(client):
    response = client.post("/api/dicom/analyze", json={"dataset_id": "ds_missing"}, params={"wait": True})
    text = json.dumps(response.json())
    assert "Traceback" not in text
    assert 'File "' not in text
    assert "site-packages" not in text
