"""Core domain models.

These are plain dataclasses (not Pydantic) because they hold NumPy arrays and
are used purely server-side.  Pydantic schemas in ``app/schemas`` handle the
HTTP boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Patient coordinate system
# ---------------------------------------------------------------------------
# DICOM patient coordinates are **LPS**:
#   +X -> patient Left
#   +Y -> patient Posterior
#   +Z -> patient Superior (head)
#
# The web viewer (glTF / three.js) is Y-up, right-handed.  We therefore expose
# meshes and points in a "viewer space" defined by
#
#   X_view =  X_lps        (patient Left  -> +X)
#   Y_view =  Z_lps        (patient Superior -> +Y, i.e. up)
#   Z_view = -Y_lps        (patient Anterior -> +Z, i.e. towards viewer)
#
# The transform is a rotation (det = +1) so it preserves handedness and volume.
VIEW_FROM_LPS = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def lps_to_view(points: np.ndarray) -> np.ndarray:
    """Convert an (..., 3) array of LPS points into viewer space."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[-1] != 3:
        raise ValueError("points must have a trailing axis of length 3")
    return pts @ VIEW_FROM_LPS.T


def view_to_lps(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    return pts @ VIEW_FROM_LPS


# ---------------------------------------------------------------------------
# DICOM
# ---------------------------------------------------------------------------

#: Attributes needed for an unambiguous spatial reconstruction.
REQUIRED_SPATIAL_TAGS = (
    "Rows",
    "Columns",
    "PixelSpacing",
    "ImagePositionPatient",
    "ImageOrientationPatient",
)


@dataclass
class SliceGeometry:
    """Everything about where one slice lives in patient space."""

    sop_instance_uid: str | None
    instance_number: int | None
    image_position_patient: np.ndarray  # (3,) LPS mm
    image_orientation_patient: np.ndarray  # (6,) row-cos(3) then col-cos(3)
    rows: int
    columns: int
    row_spacing: float  # mm, between rows (PixelSpacing[0])
    column_spacing: float  # mm, between columns (PixelSpacing[1])
    slice_thickness: float | None
    spacing_between_slices: float | None
    slice_location: float | None
    #: projection of IPP onto the series normal -- the physical sort key
    sort_key: float = 0.0

    @property
    def normal(self) -> np.ndarray:
        row = self.image_orientation_patient[:3]
        col = self.image_orientation_patient[3:]
        n = np.cross(row, col)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            raise ValueError("degenerate ImageOrientationPatient")
        return n / norm

    @property
    def pixel_spacing(self) -> tuple[float, float]:
        return (self.row_spacing, self.column_spacing)


@dataclass
class SeriesInfo:
    series_instance_uid: str
    study_instance_uid: str | None
    modality: str
    series_number: int | None = None
    series_description: str | None = None
    study_description: str | None = None
    body_part_examined: str | None = None
    manufacturer: str | None = None
    rows: int | None = None
    columns: int | None = None
    slice_count: int = 0
    #: number of files that parsed as DICOM and carried pixel data
    files_with_pixel_data: int = 0
    #: files that could not be read at all
    unreadable_files: list[str] = field(default_factory=list)
    #: geometric description derived during analysis
    orientation_label: str | None = None
    normal: np.ndarray | None = None
    slice_spacing: float | None = None
    irregular_spacing: bool = False
    has_complete_geometry: bool = False
    issues: list[str] = field(default_factory=list)


@dataclass
class DicomFileRecord:
    """Lightweight description of one file on disk."""

    path: str  # relative to the dataset root
    stored_name: str
    original_name: str
    size_bytes: int
    is_dicom: bool
    error: str | None = None


@dataclass
class DatasetRef:
    """A scanned DICOM dataset on disk."""

    dataset_id: str
    root: str
    created_at: float
    files: list[DicomFileRecord] = field(default_factory=list)
    series: dict[str, SeriesInfo] = field(default_factory=dict)
    #: series uid -> ordered slice geometry (populated during ordering)
    geometry: dict[str, list[SliceGeometry]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    #: series uid -> relative file paths, in reconstructed order
    ordered_paths: dict[str, list[str]] = field(default_factory=dict)

    @property
    def dicom_files(self) -> list[DicomFileRecord]:
        return [f for f in self.files if f.is_dicom]

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def total_bytes(self) -> int:
        return int(sum(f.size_bytes for f in self.files))


@dataclass
class DicomMetadata:
    """The subset of DICOM metadata the application cares about."""

    # patient-safe / non-identifying
    modality: str | None = None
    study_instance_uid: str | None = None
    series_instance_uid: str | None = None
    sop_instance_uid: str | None = None
    frame_of_reference_uid: str | None = None
    series_description: str | None = None
    study_description: str | None = None
    series_number: int | None = None
    body_part_examined: str | None = None
    patient_position: str | None = None
    manufacturer: str | None = None
    manufacturer_model: str | None = None
    # geometry
    rows: int | None = None
    columns: int | None = None
    pixel_spacing: tuple[float, float] | None = None
    slice_thickness: float | None = None
    spacing_between_slices: float | None = None
    image_position_patient: tuple[float, float, float] | None = None
    image_orientation_patient: tuple[float, ...] | None = None
    # rescale
    rescale_slope: float | None = None
    rescale_intercept: float | None = None
    rescale_type: str | None = None
    # pixel encoding
    bits_allocated: int | None = None
    pixel_representation: int | None = None
    photometric_interpretation: str | None = None
    samples_per_pixel: int | None = None
    window_center: float | None = None
    window_width: float | None = None
    # series wide
    default_window_center: float | None = None
    default_window_width: float | None = None
    #: any tags present that we consider spatially essential but empty
    missing_spatial_tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, tuple):
                out[key] = list(value)
            else:
                out[key] = value
        return out


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


@dataclass
class Volume:
    """A reconstructed 3D voxel volume in patient space.

    Index convention (ITK / SimpleITK compatible)::

        physical_point = origin + direction @ (index * spacing)

    with ``index = (k, j, i)`` where

    * ``k`` indexes the slice axis  (array axis 0)
    * ``j`` indexes rows            (array axis 1)
    * ``i`` indexes columns         (array axis 2)

    ``direction`` is a 3x3 matrix whose **columns** are the direction cosines of
    the three array axes expressed in LPS patient space.
    """

    volume_id: str
    array: np.ndarray  # shape (K, J, I) float32
    spacing: tuple[float, float, float]  # (dz, dy, dx) mm
    origin: np.ndarray  # (3,) LPS mm, physical position of voxel (0,0,0)
    direction: np.ndarray  # (3,3)
    modality: str
    series_instance_uid: str
    study_instance_uid: str | None = None
    series_description: str | None = None
    patient_position: str | None = None
    orientation_label: str = "unknown"
    window_center: float | None = None
    window_width: float | None = None
    is_resampled: bool = False
    is_normalized: bool = False
    intensity_units: str = "HU"  # "HU" for CT, "intensity" otherwise
    slice_positions_mm: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source_id: str | None = None  # dataset_id

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.array.shape[0]), int(self.array.shape[1]), int(self.array.shape[2]))

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.array.shape))

    @property
    def voxel_volume_mm3(self) -> float:
        return float(self.spacing[0] * self.spacing[1] * self.spacing[2])

    @property
    def physical_size_mm(self) -> tuple[float, float, float]:
        """(depth, height, width) of the volume bounding box in mm."""
        return (
            self.shape[0] * self.spacing[0],
            self.shape[1] * self.spacing[1],
            self.shape[2] * self.spacing[2],
        )

    def index_to_lps(self, index: np.ndarray) -> np.ndarray:
        idx = np.asarray(index, dtype=np.float64)
        return self.origin + (idx * np.asarray(self.spacing, dtype=np.float64)) @ self.direction.T

    def lps_to_view(self, points: np.ndarray) -> np.ndarray:
        return lps_to_view(points)

    def copy_with(self, **overrides: Any) -> "Volume":
        data = dict(
            volume_id=self.volume_id,
            array=self.array,
            spacing=self.spacing,
            origin=self.origin,
            direction=self.direction,
            modality=self.modality,
            series_instance_uid=self.series_instance_uid,
            study_instance_uid=self.study_instance_uid,
            series_description=self.series_description,
            patient_position=self.patient_position,
            orientation_label=self.orientation_label,
            window_center=self.window_center,
            window_width=self.window_width,
            is_resampled=self.is_resampled,
            is_normalized=self.is_normalized,
            intensity_units=self.intensity_units,
            slice_positions_mm=self.slice_positions_mm,
            warnings=list(self.warnings),
            notes=list(self.notes),
            source_id=self.source_id,
        )
        data.update(overrides)
        return Volume(**data)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


@dataclass
class SegmentationMask:
    mask: np.ndarray  # bool, shape (K, J, I)
    label: str
    color: str  # hex, e.g. "#e8b04b"
    engine: str
    params: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_voxels(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def empty(self) -> bool:
        return self.n_voxels == 0


@dataclass
class SegmentationResult:
    segmentation_id: str
    volume_id: str
    label: str
    color: str
    engine: str
    params: dict[str, Any]
    mask: np.ndarray
    created_at: float
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------


@dataclass
class MeshData:
    """Triangulated surface in **viewer** (Y-up) coordinates, millimetres."""

    vertices: np.ndarray  # (N, 3) float32
    faces: np.ndarray  # (M, 3) int32
    normals: np.ndarray | None = None

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])


@dataclass
class MeshAsset:
    mesh_id: str
    segmentation_id: str
    volume_id: str
    label: str
    color: str
    mesh: MeshData
    stats: dict[str, Any]
    created_at: float
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)  # format -> path on disk
