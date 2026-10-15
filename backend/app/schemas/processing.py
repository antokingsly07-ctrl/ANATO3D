"""Reconstruction, segmentation, mesh, validation and export schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .common import FindingModel, ValidationSummaryModel

Plane = Literal["axial", "coronal", "sagittal"]


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

class ReconstructionRequest(BaseModel):
    dataset_id: str
    series_instance_uid: str | None = Field(
        None, description="Series to reconstruct. Required when the upload contains more than one series."
    )
    allow_instance_number_fallback: bool = Field(
        False,
        description=(
            "Permit ordering by InstanceNumber when patient-space geometry is missing. "
            "The reconstruction is then flagged as spatially unverified."
        ),
    )
    uniform_spacing_mm: list[float] | None = Field(
        None, min_length=3, max_length=3, description="Force a uniform voxel grid [slice, row, column] in mm."
    )
    isotropic: bool = False


class VolumeGeometry(BaseModel):
    volume_id: str
    dataset_id: str | None = None
    modality: str
    series_instance_uid: str
    series_description: str | None = None
    dimensions: list[int]
    spacing_mm: list[float]
    origin_lps_mm: list[float]
    direction: list[list[float]]
    orientation: str
    patient_position: str | None = None
    physical_size_mm: list[float]
    voxel_volume_mm3: float
    intensity_units: str
    window_center: float | None = None
    window_width: float | None = None
    is_resampled: bool
    is_normalized: bool
    plane_slice_counts: dict[str, int]
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ReconstructionResponse(BaseModel):
    volume_id: str
    dataset_id: str
    report: dict[str, Any]
    geometry: VolumeGeometry
    findings: list[FindingModel] = Field(default_factory=list)
    summary: ValidationSummaryModel
    slice_preview: list[dict[str, Any]] = Field(default_factory=list)


class SliceResponse(BaseModel):
    volume_id: str
    plane: Plane
    index: int
    count: int
    rows: int
    columns: int
    row_spacing_mm: float
    column_spacing_mm: float
    position_mm: float
    position_label: str
    labels: dict[str, str]
    dtype: Literal["int16", "float32"]
    encoding: Literal["base64"] = "base64"
    data: str
    intensity_min: float
    intensity_max: float
    intensity_units: str
    window_center: float | None = None
    window_width: float | None = None
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class PreprocessRequest(BaseModel):
    volume_id: str
    normalize: Literal["none", "minmax", "zscore"] = "none"
    denoise: Literal["none", "gaussian", "median"] = "none"
    denoise_sigma: float = Field(1.0, ge=0.0, le=10.0)
    denoise_size: int = Field(3, ge=1, le=15)
    clip_min: float | None = None
    clip_max: float | None = None
    resample_target_mm: list[float] | None = Field(None, min_length=3, max_length=3)
    isotropic: bool = False
    downsample_factor: int = Field(1, ge=1, le=8)
    replace_original: bool = Field(False, description="Register the result as a new volume (never mutates the source).")


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

class SegmentationRequest(BaseModel):
    volume_id: str
    engine: str | None = Field(
        None,
        description="threshold | region_growing | ai | classical_fallback. Defaults to the preset's engine, else 'threshold'.",
    )
    preset: str | None = Field(None, description="Optional preset key; its parameters are merged first.")
    params: dict[str, Any] = Field(default_factory=dict)
    label: str | None = None
    color: str | None = None


class SegmentationResponse(BaseModel):
    segmentation_id: str
    volume_id: str
    label: str
    color: str
    engine: str
    requested_engine: str | None = None
    statistics: dict[str, Any]
    params: dict[str, Any]
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    findings: list[FindingModel] = Field(default_factory=list)
    duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

class MeshOptions(BaseModel):
    min_component_volume_mm3: float = Field(25.0, ge=0.0)
    keep_largest_n: int = Field(0, ge=0, le=64)
    smooth_iterations: int = Field(8, ge=0, le=100)
    smooth_lambda: float = Field(0.5, ge=0.0, le=1.0)
    smooth_nu: float = Field(-0.53, ge=-1.0, le=1.0)
    max_faces: int = Field(0, ge=0, description="0 disables decimation.")
    decimate_ratio: float = Field(1.0, gt=0.0, le=1.0)
    export_formats: list[Literal["glb", "stl", "obj", "ply"]] = Field(default_factory=lambda: ["glb", "stl", "obj"])


class MeshRequest(BaseModel):
    segmentation_id: str
    options: MeshOptions = Field(default_factory=MeshOptions)


class MeshResponse(BaseModel):
    mesh_id: str
    segmentation_id: str
    volume_id: str
    label: str
    color: str
    vertex_count: int
    face_count: int
    measurements: dict[str, Any]
    measurement_summary: list[dict[str, Any]]
    refinement: dict[str, Any]
    extraction: dict[str, Any]
    available_formats: list[str]
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ValidationRequest(BaseModel):
    dataset_id: str | None = None
    volume_id: str | None = None
    segmentation_id: str | None = None
    reference_segmentation_id: str | None = Field(
        None,
        description=(
            "Compare against another segmentation in this session. This measures method-to-method "
            "agreement, NOT accuracy."
        ),
    )
    ground_truth_id: str | None = Field(None, description="A previously uploaded reference mask.")


class GroundTruthUploadResponse(BaseModel):
    ground_truth_id: str
    dataset_id: str
    filename: str
    shape: list[int]
    voxel_count: int
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ValidationResponse(BaseModel):
    validation_id: str
    summary: ValidationSummaryModel
    findings: list[FindingModel]
    ground_truth: dict[str, Any]
    dataset_information: dict[str, Any]
    interpretation_notes: list[str] = Field(default_factory=list)
    disclaimer: str
    duration_s: float


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class ExportItem(BaseModel):
    key: str
    label: str
    description: str
    url: str | None = None
    available: bool = True
    size_bytes: int | None = None
    media_type: str | None = None


class ExportManifest(BaseModel):
    dataset_id: str
    items: list[ExportItem]
    privacy_note: str
