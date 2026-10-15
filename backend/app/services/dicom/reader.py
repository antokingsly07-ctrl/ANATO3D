"""Low-level DICOM file reading and dataset scanning.

Nothing in this module renames or modifies the uploaded files; it only reads
them.  A file is considered DICOM if it parses with either the standard
128-byte preamble or (for ``force=True``) contains at least one of the
identifying tags that every real-world DICOM file carries.

Filenames are never used for anything other than reporting.  In particular they
are **never** used to determine anatomical ordering.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError

from ...core.errors import DicomError
from ...core.logging_ import get_logger
from ...core.storage import new_id
from ...models import DicomFileRecord, DicomMetadata, DatasetRef, SliceGeometry

logger = get_logger("dicom.reader")

#: SOP class UIDs that describe images we can reconstruct.
IMAGE_SOP_PREFIXES = (
    "1.2.840.10008.5.1.4.1.1.2",  # CT Image Storage
    "1.2.840.10008.5.1.4.1.1.4",  # MR Image Storage
    "1.2.840.10008.5.1.4.1.1.4.1",  # Enhanced MR Image Storage
    "1.2.840.10008.5.1.4.1.1.4.3",  # Enhanced MR Color Image Storage
    "1.2.840.10008.5.1.4.1.1.2.1",  # Enhanced CT Image Storage
    "1.2.840.10008.5.1.4.1.1.7",  # Secondary Capture (some exports)
    "1.2.840.10008.5.1.4.1.1.6.1",  # Ultrasound (single frame) - read but unsupported spatially
    "1.2.840.10008.5.1.4.1.1.20",  # NM
    "1.2.840.10008.5.1.4.1.1.128",  # PET Image Storage
)

#: Minimum set of tags that identify a file as DICOM without a preamble.
IDENTIFYING_TAGS = (
    "SOPClassUID",
    "SOPInstanceUID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Modality",
    "PixelData",
    "Rows",
    "TransferSyntaxUID",
)

#: Tag name -> the order we report missing spatial metadata in.
SPATIAL_TAGS = (
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "Rows",
    "Columns",
)

#: Files larger than this are skipped outright (defensive; real DICOM slices are
#: a few MB at most).
MAX_CONSIDERED_FILE_BYTES = 1024 * 1024 * 1024


def iter_files(root: Path) -> Iterator[Path]:
    """Yield every regular file under ``root`` (depth-first, deterministic)."""
    if root.is_file():
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            candidate = Path(dirpath) / name
            try:
                if candidate.is_file() and candidate.stat().st_size > 0:
                    yield candidate
            except OSError:
                continue


def is_probably_dicom(path: Path) -> bool:
    """Cheap pre-filter: DICOM files start with the 'DICM' magic at offset 128.

    Files without a preamble (allowed by the standard) return False here but are
    still attempted with ``force=True`` during the scan.
    """
    try:
        if path.suffix.lower() == ".dcm":
            return True
        with open(path, "rb") as handle:
            head = handle.read(132)
        if len(head) >= 132 and head[128:132] == b"DICM":
            return True
    except OSError:
        return False
    return False


def read_dataset(path: Path, *, stop_before_pixels: bool = True) -> Dataset:
    """Read a DICOM file, tolerating missing preambles.

    Raises :class:`DicomError` if the file cannot be interpreted as DICOM.
    """
    attempts: list[dict[str, object]] = []
    try:
        with open(path, "rb") as handle:
            head = handle.read(132)
        has_preamble = len(head) >= 132 and head[128:132] == b"DICM"
    except OSError as exc:
        raise DicomError(f"Could not open '{path.name}'.") from exc

    if has_preamble:
        attempts.append({"force": False})
    attempts.append({"force": True})

    last_error: Exception | None = None
    for options in attempts:
        try:
            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=stop_before_pixels,
                force=bool(options["force"]),
                specific_tags=None,
            )
        except (InvalidDicomError, Exception) as exc:  # noqa: BLE001 - pydicom raises broadly
            last_error = exc
            continue
        if any(hasattr(ds, tag) or tag in ds.file_meta for tag in IDENTIFYING_TAGS):
            return ds
        last_error = DicomError("File does not contain recognisable DICOM tags.")
    raise DicomError(
        f"'{path.name}' is not a valid DICOM file.",
        detail=str(last_error) if last_error else None,
    )


def _to_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float_tuple(value: object) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        seq = list(value)  # DSfloat lists
    except TypeError:
        return None
    out = []
    for item in seq:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return None
    return tuple(out)


def extract_metadata(ds: Dataset) -> DicomMetadata:
    """Pull the metadata subset we care about out of a pydicom Dataset."""
    meta = DicomMetadata()

    meta.modality = _str(ds, "Modality")
    meta.study_instance_uid = _str(ds, "StudyInstanceUID")
    meta.series_instance_uid = _str(ds, "SeriesInstanceUID")
    meta.sop_instance_uid = _str(ds, "SOPInstanceUID")
    meta.frame_of_reference_uid = _str(ds, "FrameOfReferenceUID")
    meta.series_description = _str(ds, "SeriesDescription")
    meta.study_description = _str(ds, "StudyDescription")
    meta.series_number = _to_int(getattr(ds, "SeriesNumber", None))
    meta.body_part_examined = _str(ds, "BodyPartExamined")
    meta.patient_position = _str(ds, "PatientPosition")
    meta.manufacturer = _str(ds, "Manufacturer")
    meta.manufacturer_model = _str(ds, "ManufacturerModelName")

    meta.rows = _to_int(getattr(ds, "Rows", None))
    meta.columns = _to_int(getattr(ds, "Columns", None))
    meta.pixel_spacing = _to_float_tuple(getattr(ds, "PixelSpacing", None))
    meta.slice_thickness = _to_float(getattr(ds, "SliceThickness", None))
    meta.spacing_between_slices = _to_float(getattr(ds, "SpacingBetweenSlices", None))
    meta.image_position_patient = _to_float_tuple(getattr(ds, "ImagePositionPatient", None))
    meta.image_orientation_patient = _to_float_tuple(getattr(ds, "ImageOrientationPatient", None))

    meta.rescale_slope = _to_float(getattr(ds, "RescaleSlope", None))
    meta.rescale_intercept = _to_float(getattr(ds, "RescaleIntercept", None))
    meta.rescale_type = _str(ds, "RescaleType")

    meta.bits_allocated = _to_int(getattr(ds, "BitsAllocated", None))
    meta.pixel_representation = _to_int(getattr(ds, "PixelRepresentation", None))
    meta.photometric_interpretation = _str(ds, "PhotometricInterpretation")
    meta.samples_per_pixel = _to_int(getattr(ds, "SamplesPerPixel", None))
    meta.window_center = _to_float(getattr(ds, "WindowCenter", None))
    meta.window_width = _to_float(getattr(ds, "WindowWidth", None))
    meta.default_window_center = meta.window_center
    meta.default_window_width = meta.window_width

    missing = []
    for tag in SPATIAL_TAGS:
        if tag in ("SpacingBetweenSlices", "SliceThickness"):
            # not strictly required; only report when neither is available
            continue
        if _to_float_tuple(getattr(ds, tag, None)) is None and _to_int(getattr(ds, tag, None)) is None:
            missing.append(tag)
    meta.missing_spatial_tags = missing
    return meta


def _str(ds: Dataset, tag: str) -> str | None:
    value = getattr(ds, tag, None)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_slice_geometry(ds: Dataset, meta: DicomMetadata, *, frame_index: int = 0) -> SliceGeometry | None:
    """Build a :class:`SliceGeometry` from a dataset, or ``None`` if impossible."""
    if meta.image_position_patient is None or meta.image_orientation_patient is None:
        return None
    if meta.rows is None or meta.columns is None:
        return None
    if len(meta.image_orientation_patient) < 6:
        return None
    if meta.pixel_spacing is None or len(meta.pixel_spacing) < 2:
        # Fall back to unit spacing but keep the volume usable; a warning is
        # raised by the caller because physical measurements would be wrong.
        pixel_spacing = (1.0, 1.0)
    else:
        pixel_spacing = (float(meta.pixel_spacing[0]), float(meta.pixel_spacing[1]))

    position = np.asarray(meta.image_position_patient, dtype=np.float64)
    iop = np.asarray(meta.image_orientation_patient[:6], dtype=np.float64)
    geom = SliceGeometry(
        sop_instance_uid=meta.sop_instance_uid,
        instance_number=_to_int(ds.get("InstanceNumber")),
        image_position_patient=position,
        image_orientation_patient=iop,
        rows=int(meta.rows),
        columns=int(meta.columns),
        row_spacing=float(pixel_spacing[0]),
        column_spacing=float(pixel_spacing[1]),
        slice_thickness=meta.slice_thickness,
        spacing_between_slices=meta.spacing_between_slices,
        slice_location=_to_float(getattr(ds, "SliceLocation", None)),
    )
    if frame_index > 0:
        step = meta.spacing_between_slices or meta.slice_thickness or 0.0
        geom.image_position_patient = position + frame_index * step * geom.normal
        geom.instance_number = (geom.instance_number or 0) + frame_index
    return geom


def scan_dataset(root: Path, *, progress=None) -> DatasetRef:
    """Walk ``root`` and build a :class:`DatasetRef` describing every file.

    Non-DICOM and corrupt files are recorded but never raise.
    """
    dataset_id = new_id("ds")
    dataset = DatasetRef(dataset_id=dataset_id, root=str(root), created_at=0.0)

    candidates = list(iter_files(root))
    total = len(candidates)
    if progress:
        progress(f"Located {total} candidate file(s)")

    for index, path in enumerate(candidates):
        rel = str(path.relative_to(root)) if root.is_dir() else path.name
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > MAX_CONSIDERED_FILE_BYTES:
            dataset.files.append(
                DicomFileRecord(
                    path=rel,
                    stored_name=path.name,
                    original_name=path.name,
                    size_bytes=int(size),
                    is_dicom=False,
                    error="File is unreasonably large for a DICOM slice and was skipped.",
                )
            )
            continue

        record = DicomFileRecord(
            path=rel,
            stored_name=path.name,
            original_name=path.name,
            size_bytes=int(size),
            is_dicom=False,
        )
        try:
            ds = read_dataset(path, stop_before_pixels=True)
        except DicomError as exc:
            record.error = exc.message
        except Exception as exc:  # noqa: BLE001
            record.error = f"Could not read file: {type(exc).__name__}"
        else:
            record.is_dicom = True
            if getattr(ds, "Modality", None) is None and not hasattr(ds, "Rows"):
                # e.g. DICOMDIR or a structured report: valid DICOM, not an image
                record.error = "Valid DICOM object but contains no image pixel data."
        dataset.files.append(record)

        if progress and total and (index % 50 == 0 or index == total - 1):
            progress(f"Scanned {index + 1}/{total} file(s)")

    return dataset


def load_pixel_frames(path: Path, *, apply_rescale: bool = True) -> tuple[np.ndarray, DicomMetadata]:
    """Read pixel data as a ``(frames, rows, columns)`` array.

    For CT this applies ``RescaleSlope``/``RescaleIntercept`` producing Hounsfield
    Units (unless ``apply_rescale`` is False, in which case the stored integer
    values are returned).

    Raises :class:`DicomError` with an explanatory message when the pixel data
    cannot be decoded (e.g. a compressed transfer syntax without a decoder).
    """
    ds = read_dataset(path, stop_before_pixels=False)
    meta = extract_metadata(ds)

    if meta.samples_per_pixel not in (None, 1):
        raise DicomError(
            f"'{path.name}' contains colour pixel data ({meta.samples_per_pixel} samples per pixel), "
            "which this prototype does not support."
        )

    try:
        raw = np.asarray(ds.pixel_array)
    except Exception as exc:  # noqa: BLE001 - pydicom/gdcm raise broadly
        transfer = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
        raise DicomError(
            f"Pixel data in '{path.name}' could not be decoded.",
            detail=(
                f"Transfer syntax: {transfer}. "
                "If this is a compressed series, install a decoder such as 'pylibjpeg' "
                "or 'python-gdcm'."
                if transfer
                else str(exc)
            ),
        ) from exc

    if raw.ndim == 4:  # (frames, rows, columns, samples) -> take first sample
        raw = raw[..., 0]
    if raw.ndim == 2:
        raw = raw[None, :, :]
    if raw.ndim != 3:
        raise DicomError(f"Unsupported pixel array shape {raw.shape} in '{path.name}'.")

    if not apply_rescale:
        return raw, meta

    slope = meta.rescale_slope if meta.rescale_slope not in (None, 0) else 1.0
    intercept = meta.rescale_intercept if meta.rescale_intercept is not None else 0.0

    if raw.dtype.kind in "iu" and abs(slope - 1.0) < 1e-9 and abs(intercept) < 1e-9:
        # Values are already physical; keep the compact integer representation.
        return raw, meta
    return raw.astype(np.float32) * float(slope) + float(intercept), meta


def load_pixel_array(path: Path) -> tuple[np.ndarray, DicomMetadata]:
    """Read the first frame of a file as a 2D physical intensity array."""
    frames, meta = load_pixel_frames(path)
    return frames[0], meta


def count_frames(path: Path) -> int:
    """Return the number of frames in a file (1 for classic single-frame)."""
    try:
        ds = read_dataset(path, stop_before_pixels=True)
    except DicomError:
        return 0
    try:
        return max(1, int(ds.get("NumberOfFrames", 1)))
    except (TypeError, ValueError):
        return 1
