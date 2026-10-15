"""Synthetic DICOM phantom generator.

Purpose
-------
Lets the application be exercised end-to-end without downloading a public
dataset, and provides deterministic fixtures for the test suite.

This data is **synthetic geometry**, not a simulated medical image and not a
real patient.  Every object produced here is tagged
``SeriesDescription = SYNTHETIC PHANTOM ...`` and the patient name is
``SYNTHETIC^PHANTOM`` so it can never be mistaken for clinical data.

The phantom is a torso-like arrangement of ellipsoids/cylinders expressed in
Hounsfield units, which makes it a meaningful target for HU-threshold bone
segmentation and for validation of the geometric pipeline (volumes, surface
areas and dimensions of analytic shapes are known to within discretisation
error).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from ...core.logging_ import get_logger

logger = get_logger("synthetic.phantom")

STUDY_UID = generate_uid(prefix="1.2.826.0.1.3680043.10.999.")
SERIES_PREFIX = "1.2.826.0.1.3680043.10.999."

#: Materials present in the phantom, in Hounsfield units.
MATERIALS = {
    "air": -1000.0,
    "lung": -820.0,
    "fat": -90.0,
    "soft_tissue": 45.0,
    "muscle": 60.0,
    "spine": 850.0,
    "rib": 900.0,
    "lesion": 1250.0,
}

#: Objects whose analytic geometry is used by tests to check the pipeline.
BONE_MATERIALS = ("spine", "rib", "lesion")


@dataclass
class PhantomOptions:
    rows: int = 256
    columns: int = 256
    slices: int = 96
    row_spacing_mm: float = 1.0
    column_spacing_mm: float = 1.0
    slice_spacing_mm: float = 1.5
    body_radius_mm: float = 105.0
    body_height_mm: float = 320.0
    #: "axial" | "coronal" | "sagittal" | "oblique"
    orientation: str = "axial"
    #: obliquity of the slice normal in degrees (only used when orientation="oblique")
    oblique_degrees: float = 20.0
    #: add ±jitter to slice positions to exercise irregular-spacing handling
    irregular_spacing: bool = False
    irregular_jitter_mm: float = 0.35
    #: shuffle the on-disk order of filenames so ordering cannot rely on names
    shuffle_filenames: bool = True
    seed: int = 20240517
    #: write NIFTI/NPZ ground-truth bone mask alongside the DICOM files
    write_ground_truth: bool = True
    include_lesion: bool = True
    series_description: str = "SYNTHETIC PHANTOM - torso with spine, ribs and lungs"
    filename_prefix: str = "phantom"


@dataclass
class PhantomResult:
    directory: Path
    series_instance_uid: str
    study_instance_uid: str
    slice_count: int
    dimensions: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    orientation: str
    files: list[Path] = field(default_factory=list)
    ground_truth_path: Path | None = None
    bone_volume_mm3: float = 0.0
    notes: list[str] = field(default_factory=list)


def _basis(orientation: str, oblique_degrees: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (row_dir, column_dir, slice_normal) for the requested geometry."""
    orientation = orientation.lower()
    if orientation == "axial":
        row_dir = np.array([1.0, 0.0, 0.0])
        col_dir = np.array([0.0, 1.0, 0.0])
    elif orientation == "coronal":
        row_dir = np.array([1.0, 0.0, 0.0])
        col_dir = np.array([0.0, 0.0, -1.0])
    elif orientation == "sagittal":
        row_dir = np.array([0.0, 1.0, 0.0])
        col_dir = np.array([0.0, 0.0, -1.0])
    elif orientation == "oblique":
        angle = math.radians(oblique_degrees)
        row_dir = np.array([math.cos(angle), math.sin(angle), 0.0])
        col_dir = np.array([-math.sin(angle) * math.cos(angle), math.cos(angle) ** 2, math.sin(angle)])
        col_dir = col_dir / np.linalg.norm(col_dir)
        # make orthogonal
        col_dir = col_dir - np.dot(col_dir, row_dir) * row_dir
        col_dir = col_dir / np.linalg.norm(col_dir)
    else:
        raise ValueError(f"Unknown orientation '{orientation}'.")
    normal = np.cross(row_dir, col_dir)
    normal = normal / np.linalg.norm(normal)
    return row_dir, col_dir, normal


def _build_volume(options: PhantomOptions) -> np.ndarray:
    """Build the phantom as a (slices, rows, cols) HU array."""
    k = np.arange(options.slices)
    j = np.arange(options.rows)
    i = np.arange(options.columns)

    # index-space coordinates in mm, centred on the phantom
    z = (k - (options.slices - 1) / 2.0) * options.slice_spacing_mm
    y = (j - (options.rows - 1) / 2.0) * options.row_spacing_mm
    x = (i - (options.columns - 1) / 2.0) * options.column_spacing_mm

    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    volume = np.full((options.slices, options.rows, options.columns), MATERIALS["air"], dtype=np.float32)

    # body: elliptical cylinder along the slice axis
    body_a = options.body_radius_mm  # left-right
    body_b = options.body_radius_mm * 0.72  # anterior-posterior
    half_height = options.body_height_mm / 2.0
    body_mask = (X / body_a) ** 2 + (Y / body_b) ** 2 <= 1.0
    body_mask &= np.abs(Z) <= half_height
    volume[body_mask] = MATERIALS["soft_tissue"]

    # subcutaneous fat shell
    fat_mask = (X / (body_a * 1.02)) ** 2 + (Y / (body_b * 1.02)) ** 2 <= 1.0
    fat_mask &= ~((X / (body_a * 0.86)) ** 2 + (Y / (body_b * 0.86)) ** 2 <= 1.0)
    fat_mask &= np.abs(Z) <= half_height
    volume[fat_mask & body_mask] = MATERIALS["fat"]

    # lungs: two ellipsoids in the anterior half
    for sign in (-1.0, 1.0):
        cx = sign * 0.42 * body_a
        cy = -0.18 * body_b
        lung = ((X - cx) / (0.34 * body_a)) ** 2 + ((Y - cy) / (0.42 * body_b)) ** 2 + (
            Z / (0.42 * half_height)
        ) ** 2 <= 1.0
        volume[lung] = MATERIALS["lung"]

    # spine: posterior vertebral column as a stack of cylinders with discs
    spine_r = 0.13 * body_a
    spine_y = 0.52 * body_b
    spine = ((X / spine_r) ** 2 + ((Y - spine_y) / spine_r) ** 2 <= 1.0)
    spine &= np.abs(Z) <= half_height * 0.98
    disc = (np.abs((Z / options.slice_spacing_mm) % 12.0) < 3.0) & spine
    volume[spine & ~disc] = MATERIALS["spine"]
    volume[disc] = MATERIALS["soft_tissue"]

    # vertebral canal: a low-density tube just anterior to the spine
    canal = ((X / (spine_r * 0.42)) ** 2 + ((Y - (spine_y - spine_r * 1.35)) / (spine_r * 0.42)) ** 2 <= 1.0)
    canal &= np.abs(Z) <= half_height * 0.98
    volume[canal] = MATERIALS["air"] + 60.0

    # ribs: elliptical rings around the body wall
    rib_ring = (
        np.abs(np.sqrt((X / (body_a * 0.9)) ** 2 + (Y / (body_b * 0.9)) ** 2) - 1.0) < 0.035
    ) & (np.abs(Z) <= half_height * 0.75)
    rib_period = np.abs((Z / options.slice_spacing_mm) % 16.0) < 5.0
    volume[rib_ring & rib_period] = MATERIALS["rib"]

    # sternum
    sternum = (np.abs(X) <= 0.08 * body_a) & (np.abs(Y + body_b * 0.86) <= 0.045 * body_b)
    sternum &= np.abs(Z) <= half_height * 0.6
    volume[sternum] = MATERIALS["rib"]

    # a focal high-density lesion for a second bone-like target
    if options.include_lesion:
        lesion = (
            ((X - 0.20 * body_a) / (0.10 * body_a)) ** 2
            + ((Y - 0.10 * body_b) / (0.10 * body_b)) ** 2
            + ((Z - 0.25 * half_height) / (0.10 * half_height)) ** 2
            <= 1.0
        )
        volume[lesion] = MATERIALS["lesion"]

    return volume


def _write_slice(
    path: Path,
    image: np.ndarray,
    *,
    options: PhantomOptions,
    series_uid: str,
    study_uid: str,
    sop_uid: str,
    instance_number: int,
    position: np.ndarray,
    row_dir: np.ndarray,
    col_dir: np.ndarray,
) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.10.999.1"

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = study_uid

    # Deliberately obviously-synthetic identification.
    ds.PatientName = "SYNTHETIC^PHANTOM"
    ds.PatientID = "SYNTHETIC-PHANTOM-0001"
    ds.PatientBirthDate = ""
    ds.PatientSex = "O"
    ds.StudyID = "SYNTH"
    ds.AccessionNumber = ""
    ds.StudyDescription = "SYNTHETIC PHANTOM DATASET (software test only)"
    ds.SeriesDescription = options.series_description
    ds.Modality = "CT"
    ds.BodyPartExamined = "ABDOMEN"
    ds.PatientPosition = "HFS"
    ds.Manufacturer = "ANATO3D synthetic generator"
    ds.ManufacturerModelName = "phantom-v1"
    ds.SoftwareVersions = "1.0"

    ds.SeriesNumber = 1
    ds.InstanceNumber = instance_number
    ds.AcquisitionNumber = 1
    ds.StudyDate = "20240101"
    ds.StudyTime = "000000"
    ds.ContentDate = "20240101"
    ds.ContentTime = "000000"

    ds.Rows = int(image.shape[0])
    ds.Columns = int(image.shape[1])
    ds.PixelSpacing = [options.row_spacing_mm, options.column_spacing_mm]
    ds.SliceThickness = options.slice_spacing_mm
    ds.SpacingBetweenSlices = options.slice_spacing_mm
    ds.ImagePositionPatient = [float(v) for v in position]
    ds.ImageOrientationPatient = [float(v) for v in np.concatenate([row_dir, col_dir])]
    ds.SliceLocation = float(np.dot(position, np.cross(row_dir, col_dir)))
    ds.FrameOfReferenceUID = study_uid

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # stored values are unsigned
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0
    ds.RescaleType = "HU"
    ds.WindowCenter = 40
    ds.WindowWidth = 400

    stored = np.clip(np.rint(image + 1024.0), 0, 65535).astype(np.uint16)
    ds.PixelData = stored.tobytes()
    ds["PixelData"].is_undefined_length = False

    pydicom.dcmwrite(str(path), ds, write_like_original=False)


def generate_phantom_dataset(target_dir: Path, options: PhantomOptions | None = None) -> PhantomResult:
    """Write a complete synthetic CT series (plus a ground-truth mask) to disk."""
    options = options or PhantomOptions()
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    volume = _build_volume(options)
    row_dir, col_dir, normal = _basis(options.orientation, options.oblique_degrees)

    rng = np.random.default_rng(options.seed)
    spacing = np.full(options.slices, options.slice_spacing_mm, dtype=np.float64)
    if options.irregular_spacing and options.slices > 2:
        spacing = spacing + rng.uniform(
            -options.irregular_jitter_mm, options.irregular_jitter_mm, size=options.slices
        )
        spacing = np.maximum(spacing, 0.2)

    positions = np.cumsum(spacing) - spacing[0]
    origin = -0.5 * positions[-1] * normal if options.slices > 1 else np.zeros(3)
    # centre the phantom in patient space at a plausible abdominal location
    origin = origin + np.array([0.0, 0.0, 0.0])

    series_uid = generate_uid(prefix=SERIES_PREFIX)
    study_uid = generate_uid(prefix=SERIES_PREFIX)

    order = list(range(options.slices))
    if options.shuffle_filenames:
        shuffled = list(order)
        rng.shuffle(shuffled)
    else:
        shuffled = order

    files: list[Path] = []
    for write_index, slice_index in enumerate(shuffled):
        sop_uid = generate_uid(prefix=SERIES_PREFIX)
        position = origin + positions[slice_index] * normal
        name = f"{options.filename_prefix}_{write_index:04d}_{rng.integers(0, 1 << 30):09d}.dcm"
        path = target_dir / name
        _write_slice(
            path,
            volume[slice_index],
            options=options,
            series_uid=series_uid,
            study_uid=study_uid,
            sop_uid=sop_uid,
            instance_number=slice_index + 1,
            position=position,
            row_dir=row_dir,
            col_dir=col_dir,
        )
        files.append(path)

    bone_mask = np.isin(
        np.round(volume).astype(np.int32),
        [int(round(MATERIALS[m])) for m in BONE_MATERIALS],
    )
    bone_volume = float(bone_mask.sum()) * options.row_spacing_mm * options.column_spacing_mm * options.slice_spacing_mm

    ground_truth_path = None
    if options.write_ground_truth:
        ground_truth_path = target_dir / "ground_truth_bone_mask.npz"
        np.savez_compressed(ground_truth_path, mask=np.packbits(bone_mask.reshape(-1)))

    notes = [
        "Synthetic phantom data. It is a geometric test object, not a real or simulated patient.",
        f"Bone-like volume (spine + ribs + sternum{', lesion' if options.include_lesion else ''}) "
        f"= {bone_volume / 1000.0:.3f} cm3 by construction.",
        "Filenames are intentionally shuffled: correct ordering can only be achieved from spatial metadata.",
    ]
    if options.irregular_spacing:
        notes.append("Slice spacing is intentionally irregular to exercise resampling.")

    return PhantomResult(
        directory=target_dir,
        series_instance_uid=series_uid,
        study_instance_uid=study_uid,
        slice_count=options.slices,
        dimensions=(options.slices, options.rows, options.columns),
        spacing_mm=(options.slice_spacing_mm, options.row_spacing_mm, options.column_spacing_mm),
        orientation=options.orientation,
        files=files,
        ground_truth_path=ground_truth_path,
        bone_volume_mm3=bone_volume,
        notes=notes,
    )


def write_phantom_series(target_dir: Path, **kwargs) -> PhantomResult:
    """Convenience wrapper accepting keyword overrides."""
    return generate_phantom_dataset(target_dir, PhantomOptions(**kwargs))
