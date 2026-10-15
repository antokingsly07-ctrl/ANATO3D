"""Ground-truth (reference mask) loading.

Accepted formats
----------------
``.npz``   an ANATO3D mask archive: ``mask`` (bit-packed ``uint8``) plus ``shape``
``.npy``   a boolean/integer array
``.nii`` / ``.nii.gz``  NIfTI (requires ``nibabel``)
``.dcm`` / no extension  a DICOM series folder or a single DICOM mask slice set
           (a zip archive of a DICOM series is also accepted)

Anything else is rejected with a clear message.  Shape mismatches are resolved by
nearest-neighbour resampling, and that is always reported.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...core.errors import UnsupportedFileError, ValidationError
from ...core.logging_ import get_logger
from ...core.security import sanitize_filename

logger = get_logger("validation.ground_truth")

NIFTI_SUFFIXES = (".nii", ".nii.gz")


@dataclass
class GroundTruth:
    mask: np.ndarray
    source: str
    shape: tuple[int, ...]
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _load_npz(path: Path) -> GroundTruth:
    notes: list[str] = []
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        if "array" in keys:
            array = np.asarray(data["array"])
        elif "mask" in keys:
            stored = data["mask"]
            if "shape" in keys:
                shape = tuple(int(v) for v in data["shape"])
                array = np.unpackbits(stored)[: int(np.prod(shape))].astype(bool).reshape(shape)
                notes.append(f"Bit-packed mask unpacked to shape {shape}.")
            else:
                array = stored.astype(bool)
                notes.append("Bit-packed mask unpacked using its own length (no stored shape).")
        else:
            raise ValidationError(
                "The mask archive does not contain an 'array', 'mask' or 'shape' entry.",
                detail=f"Found entries: {', '.join(sorted(keys))}",
            )
    return GroundTruth(mask=array.astype(bool), source=path.name, shape=array.shape, notes=notes)


def _load_npy(path: Path) -> GroundTruth:
    array = np.load(path, allow_pickle=False)
    return GroundTruth(mask=np.asarray(array).astype(bool), source=path.name, shape=np.asarray(array).shape)


def _load_nifti(path: Path) -> GroundTruth:
    try:
        import nibabel as nib
    except Exception as exc:  # noqa: BLE001
        raise UnsupportedFileError(
            "NIfTI ground-truth masks require the 'nibabel' package.",
            detail=f"pip install nibabel ({exc})",
        ) from exc
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValidationError(
            f"NIfTI ground-truth mask must be 3D; '{path.name}' has {data.ndim} dimensions ({data.shape})."
        )
    # NIfTI is radiological (X varies fastest); our volumes are (k, j, i).
    array = np.transpose(data, (2, 1, 0))
    notes = [
        f"NIfTI loaded ({data.shape}) and transposed to slice x row x column ordering.",
    ]
    return GroundTruth(mask=array > 0.5, source=path.name, shape=array.shape, notes=notes)


def _load_dicom_mask(path: Path) -> GroundTruth:
    """Load a mask expressed as a DICOM series (single-level) or a zip of one."""
    import tempfile

    from ..dicom import reader, series as series_service

    cleanup: Path | None = None
    root = path
    if path.suffix.lower() == ".zip":
        cleanup = Path(tempfile.mkdtemp(prefix="anato3d_gt_"))
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                target_name = sanitize_filename(member)
                if not target_name:
                    continue
                (cleanup / target_name).write_bytes(archive.read(member))
        root = cleanup

    try:
        dataset = reader.scan_dataset(root)
        analysis = series_service.analyze_dataset(dataset, root)
        if not analysis.series:
            raise ValidationError("The ground-truth archive contains no readable DICOM series.")
        entry = analysis.series[0]
        from ..reconstruction.volume import build_volume

        result = build_volume(dataset, root, entry)
        array = np.asarray(result.volume.array)
        mask = array > 0
        return GroundTruth(
            mask=mask,
            source=path.name,
            shape=mask.shape,
            notes=[f"DICOM ground-truth series reconstructed ({mask.shape}); non-zero voxels treated as foreground."],
        )
    finally:
        if cleanup is not None:
            import shutil

            shutil.rmtree(cleanup, ignore_errors=True)


def load_ground_truth(path: Path) -> GroundTruth:
    name = path.name.lower()
    if name.endswith(".zip"):
        return _load_dicom_mask(path)
    if name.endswith(".npz"):
        return _load_npz(path)
    if name.endswith(".npy"):
        return _load_npy(path)
    if name.endswith(NIFTI_SUFFIXES):
        return _load_nifti(path)
    if path.is_dir() or name.endswith(".dcm"):
        return _load_dicom_mask(path)
    raise UnsupportedFileError(
        f"Unsupported ground-truth format for '{path.name}'.",
        detail="Supported: .npz (ANATO3D mask), .npy, .nii/.nii.gz (NIfTI), .zip or a DICOM series.",
    )


def align_to_shape(
    ground_truth: GroundTruth,
    target_shape: tuple[int, int, int],
) -> GroundTruth:
    """Nearest-neighbour resample a reference mask onto the volume grid."""
    mask = ground_truth.mask
    if tuple(mask.shape) == tuple(target_shape):
        return ground_truth

    if mask.ndim != 3:
        raise ValidationError(
            "Ground-truth mask must be 3D to be compared against a reconstructed volume.",
            detail=f"Got shape {mask.shape}.",
        )

    factors = [target_shape[a] / mask.shape[a] for a in range(3)]
    from scipy import ndimage

    resampled = ndimage.zoom(mask.astype(np.uint8), factors, order=0) > 0
    # zoom can be off-by-one because of rounding; pad/crop to the exact target
    resampled = _fit(transformed=resampled, target_shape=target_shape)
    warnings = list(ground_truth.warnings)
    warnings.append(
        "Ground-truth mask shape {} did not match the volume {}; it was resampled with nearest-"
        "neighbour interpolation. Metrics carry an additional resampling error.".format(
            tuple(mask.shape), tuple(target_shape)
        )
    )
    return GroundTruth(
        mask=resampled,
        source=ground_truth.source,
        shape=resampled.shape,
        notes=list(ground_truth.notes),
        warnings=warnings,
    )


def _fit(*, transformed: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    result = transformed
    crop = tuple(slice(0, min(result.shape[a], target_shape[a])) for a in range(3))
    result = result[crop]
    pad = tuple((0, max(0, target_shape[a] - result.shape[a])) for a in range(3))
    if any(p[1] > 0 for p in pad):
        result = np.pad(result, pad, mode="constant", constant_values=False)
    return result.astype(bool)
