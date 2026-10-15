"""End-to-end HTTP smoke test driving the full pipeline through the JSON API.

Run from the ``backend`` directory::

    python scripts/smoke_api.py

It uses a temporary ``ANATO3D_DATA_DIR`` so it never touches real data.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("ANATO3D_DATA_DIR", tempfile.mkdtemp(prefix="anato3d_api_"))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def check(condition: bool, message: str) -> None:
    status = "OK " if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        raise SystemExit(f"Smoke test failed: {message}")


def unwrap(response, label: str) -> dict:
    check(response.status_code == 200, f"{label}: HTTP {response.status_code}")
    payload = response.json()
    check(payload.get("status") == "completed", f"{label}: job status {payload.get('status')} ({payload.get('error')})")
    result = payload.get("result")
    check(isinstance(result, dict), f"{label}: produced a result object")
    return result


def main() -> None:
    started = time.time()

    # 1. Synthetic dataset ---------------------------------------------------
    synthetic = client.post(
        "/api/dicom/synthetic",
        json={
            "rows": 64,
            "columns": 64,
            "slices": 32,
            "row_spacing_mm": 1.5,
            "column_spacing_mm": 1.5,
            "slice_spacing_mm": 2.0,
            "orientation": "axial",
        },
    )
    check(synthetic.status_code == 200, f"synthetic: HTTP {synthetic.status_code}")
    synthetic_body = synthetic.json()
    dataset_id = synthetic_body["dataset_id"]
    check(synthetic_body["label"] == "SYNTHETIC", "synthetic: labelled SYNTHETIC")
    check(synthetic_body["ground_truth_available"], "synthetic: ground truth present")
    ground_truth_id = synthetic_body["ground_truth_id"]
    print(f"     dataset_id={dataset_id} gt={ground_truth_id}")

    # 2. Analyse -------------------------------------------------------------
    analysis = unwrap(
        client.post("/api/dicom/analyze", json={"dataset_id": dataset_id}, params={"wait": True}),
        "analyze",
    )
    check(len(analysis["series"]) >= 1, "analyze: at least one series found")
    check(len(analysis["ordering_preview"]) == 32, "analyze: ordering preview has all slices")
    series_uid = analysis["primary_series_uid"]
    check(bool(series_uid), "analyze: primary series selected")
    check(analysis["summary"]["status"] in ("ok", "warning", "error"), "analyze: summary status present")

    # Preview ordering must be monotonic in patient-space projection.
    projections = [entry["projection_mm"] for entry in analysis["ordering_preview"]]
    check(
        all(projections[i] <= projections[i + 1] + 1e-6 for i in range(len(projections) - 1)),
        "analyze: slices ordered monotonically by patient position",
    )

    # 3. Reconstruct ---------------------------------------------------------
    recon = unwrap(
        client.post(
            "/api/reconstruction/create",
            json={"dataset_id": dataset_id, "series_instance_uid": series_uid},
            params={"wait": True},
        ),
        "reconstruct",
    )
    volume_id = recon["volume_id"]
    geometry = recon["geometry"]
    check(geometry["dimensions"] == [32, 64, 64], f"reconstruct: dimensions {geometry['dimensions']}")
    check(abs(geometry["spacing_mm"][0] - 2.0) < 1e-6, "reconstruct: slice spacing preserved")
    check(len(geometry["direction"]) == 3, "reconstruct: direction matrix present")
    check(len(recon["slice_preview"]) == 3, "reconstruct: preview rendered for 3 planes")

    # 4. Slice + thumbnail ---------------------------------------------------
    slice_resp = client.get(
        f"/api/reconstruction/{volume_id}/slice",
        params={"plane": "axial", "index": 16},
    )
    check(slice_resp.status_code == 200, "slice: HTTP 200")
    slice_body = slice_resp.json()
    raw = base64.b64decode(slice_body["data"])
    itemsize = 2 if slice_body["dtype"] == "int16" else 4
    check(
        len(raw) == slice_body["rows"] * slice_body["columns"] * itemsize,
        f"slice: payload size matches rows*cols*{itemsize} ({slice_body['dtype']})",
    )
    check(slice_body["position_label"] != "", "slice: has an anatomical position label")

    thumb = client.get(f"/api/reconstruction/{volume_id}/thumbnail", params={"plane": "sagittal"})
    check(thumb.status_code == 200, "thumbnail: HTTP 200")
    check(thumb.content[:8] == b"\x89PNG\r\n\x1a\n", "thumbnail: valid PNG signature")

    # 5. Segment -------------------------------------------------------------
    segmentation = unwrap(
        client.post(
            "/api/segmentation/run",
            json={"volume_id": volume_id, "engine": "threshold", "preset": "ct_bone"},
            params={"wait": True},
        ),
        "segment",
    )
    segmentation_id = segmentation["segmentation_id"]
    check(segmentation["statistics"]["voxel_count"] > 0, "segment: non-empty mask")
    print(
        f"     mask: {segmentation['statistics']['voxel_count']} voxels, "
        f"{segmentation['statistics']['volume_cm3']} cm3"
    )

    mask_resp = client.get(
        f"/api/segmentation/{segmentation_id}/mask", params={"plane": "axial", "index": 16}
    )
    check(mask_resp.status_code == 200, "mask slice: HTTP 200")
    check(mask_resp.json()["encoding"] == "packbits-base64", "mask slice: bit-packed")

    # 6. Mesh ----------------------------------------------------------------
    mesh = unwrap(
        client.post("/api/mesh/generate", json={"segmentation_id": segmentation_id}, params={"wait": True}),
        "mesh",
    )
    mesh_id = mesh["mesh_id"]
    check(mesh["face_count"] > 0, "mesh: has triangles")
    check("glb" in mesh["available_formats"], "mesh: GLB exported")

    glb = client.get(f"/api/mesh/{mesh_id}/glb")
    check(glb.status_code == 200, "mesh glb: HTTP 200")
    check(glb.content[:4] == b"glTF", "mesh glb: valid glTF magic")

    # 7. Analysis ------------------------------------------------------------
    analysis_payload = client.get(f"/api/analysis/{mesh_id}")
    check(analysis_payload.status_code == 200, "analysis: HTTP 200")
    analysis_body = analysis_payload.json()
    voxel_volume = analysis_body["voxel"]["volume_mm3"]
    check(voxel_volume > 0, "analysis: voxel volume computed")
    print(f"     voxel volume = {voxel_volume / 1000:.3f} cm3")

    # 8. Validation against the known synthetic ground truth -----------------
    validation = unwrap(
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
    validation_id = validation["validation_id"]
    gt_block = validation["ground_truth"]
    check(gt_block["state"] == "computed", "validation: ground-truth metrics computed")
    metrics = {m["key"]: m["value"] for m in gt_block.get("metrics", [])}
    check("dice" in metrics, "validation: Dice reported")
    print(f"     dice={metrics.get('dice')} iou={metrics.get('iou')} hd95={metrics.get('hausdorff95')}")

    # 9. Export --------------------------------------------------------------
    manifest = client.get(
        f"/api/export/{dataset_id}/manifest",
        params={
            "volume_id": volume_id,
            "segmentation_id": segmentation_id,
            "mesh_id": mesh_id,
            "validation_id": validation_id,
        },
    )
    check(manifest.status_code == 200, "export manifest: HTTP 200")
    items = {item["key"]: item for item in manifest.json()["items"]}
    check(items["volume_nifti"]["available"], "export manifest: volume available")
    check(items["bundle"]["available"], "export manifest: bundle available")

    for key in ("volume_nifti", "mask_npz", "analysis_json", "validation_json"):
        resp = client.get(
            f"/api/export/{dataset_id}/file/{key}",
            params={
                "volume_id": volume_id,
                "segmentation_id": segmentation_id,
                "mesh_id": mesh_id,
                "validation_id": validation_id,
            },
        )
        check(resp.status_code == 200, f"export {key}: HTTP 200")
        check(len(resp.content) > 0, f"export {key}: non-empty")

    bundle = client.get(
        f"/api/export/{dataset_id}/file/bundle",
        params={"volume_id": volume_id, "segmentation_id": segmentation_id, "mesh_id": mesh_id},
    )
    check(bundle.status_code == 200, "export bundle: HTTP 200")
    check(bundle.content[:2] == b"PK", "export bundle: valid zip")

    # 10. Jobs endpoint ------------------------------------------------------
    jobs = client.get("/api/jobs")
    check(jobs.status_code == 200 and len(jobs.json()) >= 5, "jobs: list populated")

    print(f"\nAll API checks passed in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
