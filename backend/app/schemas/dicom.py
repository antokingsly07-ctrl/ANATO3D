"""DICOM endpoints schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .common import FindingModel, ValidationSummaryModel


class SeriesSummary(BaseModel):
    series_instance_uid: str
    study_instance_uid: str | None = None
    modality: str
    series_number: int | None = None
    series_description: str | None = None
    body_part_examined: str | None = None
    rows: int | None = None
    columns: int | None = None
    slice_count: int
    orientation: str | None = None
    slice_spacing_mm: float | None = None
    irregular_spacing: bool = False
    has_complete_geometry: bool = False
    is_reconstructable: bool = False
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class StudySummary(BaseModel):
    study_instance_uid: str
    series_count: int
    slice_count: int
    study_description: str | None = None
    modalities: list[str] = Field(default_factory=list)
    series_instance_uids: list[str] = Field(default_factory=list)


class DatasetSummary(BaseModel):
    dataset_id: str
    file_count: int
    dicom_file_count: int
    non_image_file_count: int = 0
    unreadable_file_count: int = 0
    total_bytes: int
    is_synthetic: bool = False
    synthetic_notes: list[str] = Field(default_factory=list)
    files: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-file scan results (name, size, whether DICOM, parse error if any).",
    )
    warnings: list[str] = Field(default_factory=list)


class MetadataReport(BaseModel):
    """Non-identifying metadata actually used by the pipeline."""

    modality: str | None = None
    study_instance_uid: str | None = None
    series_instance_uid: str | None = None
    series_description: str | None = None
    study_description: str | None = None
    series_number: int | None = None
    body_part_examined: str | None = None
    patient_position: str | None = None
    manufacturer: str | None = None
    rows: int | None = None
    columns: int | None = None
    pixel_spacing_mm: list[float] | None = None
    slice_thickness_mm: float | None = None
    spacing_between_slices_mm: float | None = None
    image_position_patient_mm: list[float] | None = None
    image_orientation_patient: list[float] | None = None
    rescale_slope: float | None = None
    rescale_intercept: float | None = None
    window_center: float | None = None
    window_width: float | None = None
    missing_spatial_tags: list[str] = Field(default_factory=list)


class SliceOrderEntry(BaseModel):
    index: int
    sop_instance_uid: str | None = None
    instance_number: int | None = None
    position_mm: list[float] | None = None
    projection_mm: float
    file: str | None = None


class AnalyzeResponse(BaseModel):
    dataset_id: str
    studies: list[StudySummary]
    series: list[SeriesSummary]
    primary_series_uid: str | None = None
    metadata_report: dict[str, Any] = Field(
        default_factory=dict, description="Metadata of the first slice of the primary series."
    )
    findings: list[FindingModel] = Field(default_factory=list)
    summary: ValidationSummaryModel
    ordering_preview: list[SliceOrderEntry] = Field(
        default_factory=list, description="First 500 slices of the primary series in reconstruction order."
    )
    frame_description: dict[str, str] = Field(default_factory=dict)
    non_image_files: int = 0
    unreadable_files: int = 0
    elapsed_s: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    dataset_id: str = Field(..., description="Id returned by the upload or synthetic endpoint.")


class SyntheticRequest(BaseModel):
    rows: int = Field(256, ge=32, le=1024)
    columns: int = Field(256, ge=32, le=1024)
    slices: int = Field(96, ge=8, le=1024)
    row_spacing_mm: float = Field(1.0, gt=0.01, le=10.0)
    column_spacing_mm: float = Field(1.0, gt=0.01, le=10.0)
    slice_spacing_mm: float = Field(1.5, gt=0.01, le=20.0)
    orientation: Literal["axial", "coronal", "sagittal", "oblique"] = "axial"
    oblique_degrees: float = Field(20.0, ge=1.0, le=45.0)
    irregular_spacing: bool = False
    include_lesion: bool = True
    shuffle_filenames: bool = True


class SyntheticResponse(BaseModel):
    dataset_id: str
    slice_count: int
    dimensions: list[int]
    spacing_mm: list[float]
    orientation: str
    ground_truth_available: bool
    ground_truth_id: str | None = None
    bone_volume_cm3: float
    notes: list[str]
    label: Literal["SYNTHETIC"] = "SYNTHETIC"
