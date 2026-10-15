"""Mesh generation and download endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from ...core.errors import NotFoundError
from ...core.logging_ import get_logger
from ...core.storage import store
from ...models.records import AnalysisRecord, MeshRecord
from ...schemas.common import JobEnvelope
from ...schemas.processing import MeshRequest
from ...services.mesh.refine import MeshRefinementOptions
from ...services.mesh.service import generate_mesh
from .. import deps
from .. import presenters

logger = get_logger("api.mesh")

router = APIRouter(prefix="/mesh", tags=["mesh"])

_MEDIA_TYPES = {
    "glb": "model/gltf-binary",
    "gltf": "model/gltf+json",
    "stl": "model/stl",
    "obj": "text/plain",
    "ply": "application/octet-stream",
}


@router.post("/generate", response_model=JobEnvelope, summary="Generate a 3D mesh from a segmentation")
def generate(
    request: MeshRequest,
    wait: bool = Query(False, description="Run inline and return the result instead of a job id."),
) -> JobEnvelope:
    segmentation = deps.get_segmentation(request.segmentation_id)
    volume_record = deps.get_volume(segmentation.result.volume_id)
    dataset_id = volume_record.dataset_id
    workspace = store.workspace(dataset_id)
    output_dir = workspace.derived_dir / "meshes"

    options = MeshRefinementOptions(
        min_component_volume_mm3=request.options.min_component_volume_mm3,
        keep_largest_n=request.options.keep_largest_n,
        smooth_iterations=request.options.smooth_iterations,
        smooth_lambda=request.options.smooth_lambda,
        smooth_nu=request.options.smooth_nu,
        max_faces=request.options.max_faces,
        decimate_ratio=request.options.decimate_ratio,
    )
    export_formats = tuple(request.options.export_formats)

    def job(reporter):
        reporter.step("mesh_generation", 0.01, "Extracting isosurface (Marching Cubes)")
        asset = generate_mesh(
            segmentation.result,
            volume_record.volume,
            output_dir=output_dir,
            options=options,
            export_formats=export_formats,
            progress=reporter,
        )
        record = MeshRecord(dataset_id=dataset_id, asset=asset)
        deps.register_mesh(record)

        payload = presenters.analysis_payload(asset, volume_record.volume, segmentation.result)
        deps.register_analysis(asset.mesh_id, AnalysisRecord(dataset_id=dataset_id, payload=payload))
        deps.register_analysis(asset.segmentation_id, AnalysisRecord(dataset_id=dataset_id, payload=payload))

        reporter.step("done", 1.0, f"Mesh ready: {asset.stats['face_count']} triangles")
        return presenters.mesh_response(asset)

    return deps.dispatch("mesh", job, wait=wait)


@router.get("/{mesh_id}", summary="Retrieve mesh metadata")
def get_mesh(mesh_id: str) -> dict:
    return presenters.mesh_response(deps.get_mesh(mesh_id).asset)


@router.get("/{mesh_id}/glb", summary="Download the GLB model")
def get_glb(mesh_id: str) -> FileResponse:
    path = _format_path(mesh_id, "glb")
    return FileResponse(path, media_type=_MEDIA_TYPES["glb"], filename=f"{mesh_id}.glb")


@router.get("/{mesh_id}/download", summary="Download the mesh in a chosen format")
def download(mesh_id: str, format: str = Query("glb")) -> FileResponse:
    requested = format.lower()
    if requested not in _MEDIA_TYPES:
        requested = "glb"
    path = _format_path(mesh_id, requested)
    return FileResponse(
        path,
        media_type=_MEDIA_TYPES.get(requested, "application/octet-stream"),
        filename=f"{mesh_id}.{requested}",
    )


def _format_path(mesh_id: str, fmt: str):
    asset = deps.get_mesh(mesh_id).asset
    path = asset.files.get(fmt)
    if not path:
        raise NotFoundError(
            f"The mesh was not exported as '{fmt}'.",
            detail=f"Available formats: {', '.join(sorted(asset.files))}.",
        )
    from pathlib import Path

    resolved = Path(path)
    if not resolved.exists():
        raise NotFoundError("The mesh file is no longer on disk. Generate the mesh again.")
    return resolved


__all__ = ["router"]
