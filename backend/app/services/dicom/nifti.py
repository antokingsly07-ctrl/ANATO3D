"""Optional NIfTI (.nii / .nii.gz) ingestion.

A user may upload a single-file volume instead of a DICOM series.  Rather than
rejecting it, we convert it into an equivalent DICOM CT series so the rest of the
pipeline (scanning, analysis, reconstruction) is unchanged.

Conversion preserves the physical geometry encoded in the NIfTI affine: voxel
spacing, slice direction and origin are translated into ``PixelSpacing``,
``ImageOrientationPatient`` and ``ImagePositionPatient``.  Intensities are
stored as unsigned 16-bit values with a ``RescaleSlope``/``RescaleIntercept`` so that
``stored * RescaleSlope + RescaleIntercept`` recovers the original values. Wide
intensity ranges are compressed with a larger slope rather than clamped.

This is a best-effort convenience for research use.  It is not a validated
medical conversion and produces no diagnostic-quality DICOM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...core.errors import DicomError
from ...core.logging_ import get_logger

logger = get_logger("dicom.nifti")

#: Filename suffixes treated as NIfTI volumes (checked longest-first).
NIFTI_SUFFIXES = (".nii.gz", ".nii")

UID_PREFIX = "1.2.826.0.1.3680043.10.777."


def is_nifti(path: Path) -> bool:
    """True if ``path`` looks like a NIfTI volume by extension."""
    name = path.name.lower()
    return name.endswith(".nii.gz") or name.endswith(".nii")


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - nibabel is a declared dependency
        raise DicomError(
            "NIfTI volumes require the 'nibabel' package.",
            detail=f"Ask the operator to pip install nibabel ({exc}).",
        ) from exc

    try:
        image = nib.load(str(path))
    except Exception as exc:  # noqa: BLE001 - nibabel raises broadly
        raise DicomError(f"'{path.name}' could not be read as a NIfTI volume.", detail=str(exc)) from exc

    data = np.asanyarray(image.dataobj)
    # Drop any leading/trailing singleton dimensions.
    data = np.squeeze(data)
    # 4-D (vector / time-series) volumes: keep the first 3-D frame.
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise DicomError(
            f"'{path.name}' has {data.ndim} dimensions; a 3-D volume is required.",
            detail=f"Array shape: {tuple(data.shape)}.",
        )
    if min(data.shape) < 1:
        raise DicomError(f"'{path.name}' contains an empty dimension.")

    affine = np.array(image.affine, dtype=np.float64)
    return data, affine


def _unit(vector: np.ndarray) -> tuple[np.ndarray, float]:
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        raise DicomError("The NIfTI affine has a degenerate (zero-length) axis.")
    return vector / length, length


def _write_slice(
    path: Path,
    image: np.ndarray,
    *,
    stored_dtype: np.ndarray,
    intercept: float,
    slope: float,
    series_uid: str,
    study_uid: str,
    sop_uid: str,
    instance_number: int,
    position: np.ndarray,
    row_cosines: np.ndarray,
    col_cosines: np.ndarray,
    pixel_spacing: list[float],
    slice_thickness: float,
    window_center: float,
    window_width: float,
) -> None:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.10.777.1"

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = study_uid

    # Clearly labelled as an imported, converted volume.
    ds.PatientName = "ANATO3D^IMPORTED"
    ds.PatientID = "ANATO3D-IMPORT"
    ds.PatientBirthDate = ""
    ds.PatientSex = "O"
    ds.StudyID = "IMPORT"
    ds.AccessionNumber = ""
    ds.StudyDescription = "Volume imported from NIfTI and converted to DICOM"
    ds.SeriesDescription = "CONVERTED FROM NIFTI"
    ds.Modality = "CT"
    ds.PatientPosition = "HFS"
    ds.Manufacturer = "ANATO3D importer"
    ds.ManufacturerModelName = "nifti-to-dicom"

    ds.SeriesNumber = 1
    ds.InstanceNumber = instance_number
    ds.AcquisitionNumber = 1
    ds.StudyDate = "19700101"
    ds.StudyTime = "000000"
    ds.ContentDate = "19700101"
    ds.ContentTime = "000000"

    ds.Rows = int(image.shape[0])
    ds.Columns = int(image.shape[1])
    ds.PixelSpacing = [float(v) for v in pixel_spacing]
    ds.SliceThickness = float(slice_thickness)
    ds.SpacingBetweenSlices = float(slice_thickness)
    ds.ImagePositionPatient = [float(v) for v in position]
    ds.ImageOrientationPatient = [float(v) for v in np.concatenate([row_cosines, col_cosines])]
    ds.SliceLocation = float(np.dot(position, np.cross(row_cosines, col_cosines)))

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # stored values are unsigned
    ds.RescaleSlope = float(slope)
    ds.RescaleIntercept = float(intercept)
    ds.RescaleType = "US"
    ds.WindowCenter = float(window_center)
    ds.WindowWidth = float(window_width)

    ds.PixelData = stored_dtype.tobytes()
    ds["PixelData"].is_undefined_length = False

    pydicom.dcmwrite(str(path), ds, write_like_original=False)


def convert_nifti_series(path: Path, out_dir: Path) -> list[Path]:
    """Convert a NIfTI volume into a DICOM CT series written under ``out_dir``.

    Returns the list of written DICOM files.  Raises :class:`DicomError` when the
    volume cannot be interpreted.
    """
    from pydicom.uid import generate_uid

    data, affine = _load(path)

    axis0, spacing0 = _unit(affine[:3, 0])  # maps to image rows
    axis1, spacing1 = _unit(affine[:3, 1])  # maps to image columns
    axis2, spacing2 = _unit(affine[:3, 2])  # slice stacking direction
    origin = affine[:3, 3]

    # image[row, col] = volume[row, col, slice]; DICOM "row direction" follows
    # the column axis of the image and vice-versa. Slice count drives the loop.
    slices = int(data.shape[2])

    data_f = data.astype(np.float32, copy=False)
    vmin = float(np.min(data_f))
    vmax = float(np.max(data_f))
    span = vmax - vmin
    # Store as unsigned 16-bit. A unit slope is exact whenever the range fits;
    # otherwise compress so no values are clipped.
    slope = 1.0 if span <= 65535.0 else span / 65535.0
    intercept = vmin
    stored = np.clip(np.rint((data_f - vmin) / slope), 0, 65535).astype(np.uint16)

    window_center = (vmin + vmax) / 2.0
    window_width = max(vmax - vmin, 1.0)

    series_uid = generate_uid(prefix=UID_PREFIX)
    study_uid = generate_uid(prefix=UID_PREFIX)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for index in range(slices):
        sop_uid = generate_uid(prefix=UID_PREFIX)
        position = origin + index * axis2 * spacing2
        target = out_dir / f"slice_{index:04d}.dcm"
        _write_slice(
            target,
            data[:, :, index],
            stored_dtype=stored[:, :, index],
            intercept=intercept,
            slope=slope,
            series_uid=series_uid,
            study_uid=study_uid,
            sop_uid=sop_uid,
            instance_number=index + 1,
            position=position,
            row_cosines=axis1,
            col_cosines=axis0,
            pixel_spacing=[spacing0, spacing1],
            slice_thickness=spacing2,
            window_center=window_center,
            window_width=window_width,
        )
        written.append(target)

    logger.info("Converted %s -> %d DICOM slice(s) (spacing %.3f/%.3f/%.3f mm)", path.name, slices, spacing2, spacing0, spacing1)
    return written
