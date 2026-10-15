"""Configurable preprocessing.

Everything here is deterministic and reversible in the sense that the *source*
volume is never mutated: each operation returns a new :class:`Volume`.  For CT
the pipeline works in Hounsfield Units, so threshold-based segmentation keeps a
physical meaning.  Normalisation is opt-in and, when enabled, is flagged on the
volume so that HU presets can be disabled in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from ...core.errors import PreprocessingError
from ...core.logging_ import get_logger
from ...models import Volume
from .volume import resample_volume

logger = get_logger("reconstruction.preprocessing")


@dataclass(frozen=True)
class WindowPreset:
    key: str
    label: str
    center: float
    width: float
    modalities: tuple[str, ...]
    description: str


#: Clinical window presets.  ``modalities`` documents which modalities the
#: preset is meaningful for -- presets are never applied automatically to MRI.
WINDOW_PRESETS: tuple[WindowPreset, ...] = (
    WindowPreset("ct_bone", "Bone", 300.0, 1500.0, ("CT", "CBCT"), "Cortical and trabecular bone."),
    WindowPreset("ct_soft_tissue", "Soft tissue", 40.0, 400.0, ("CT", "CBCT"), "General soft tissue."),
    WindowPreset("ct_lung", "Lung", -600.0, 1500.0, ("CT", "CBCT"), "Pulmonary parenchyma."),
    WindowPreset("ct_brain", "Brain", 40.0, 80.0, ("CT", "CBCT"), "Intracranial contents."),
    WindowPreset("ct_abdomen", "Abdomen", 60.0, 400.0, ("CT", "CBCT"), "Abdominal organs."),
    WindowPreset("ct_mediastinum", "Mediastinum", 50.0, 350.0, ("CT", "CBCT"), "Mediastinal soft tissue."),
    WindowPreset("ct_angio", "Angiography", 300.0, 600.0, ("CT", "CBCT"), "Contrast-enhanced vessels."),
)

#: Nominal HU ranges used to seed threshold segmentation presets.
HU_REFERENCE = {
    "bone": (200.0, 3071.0),
    "soft_tissue": (-150.0, 250.0),
    "lung": (-1000.0, -400.0),
    "air": (-1100.0, -900.0),
    "fat": (-150.0, -30.0),
}


@dataclass
class PreprocessingOptions:
    normalize: str = "none"  # none | minmax | zscore
    denoise: str = "none"  # none | gaussian | median
    denoise_sigma: float = 1.0
    denoise_size: int = 3
    clip_min: float | None = None
    clip_max: float | None = None
    resample_target_mm: tuple[float, float, float] | None = None
    isotropic: bool = False
    downsample_factor: int = 1
    metadata: dict[str, object] = field(default_factory=dict)


def presets_for_modality(modality: str | None) -> list[dict[str, object]]:
    """Window presets relevant to a modality, plus a note for non-CT data."""
    mod = (modality or "").upper()
    out: list[dict[str, object]] = []
    for preset in WINDOW_PRESETS:
        out.append(
            {
                "key": preset.key,
                "label": preset.label,
                "center": preset.center,
                "width": preset.width,
                "description": preset.description,
                "applicable": mod in preset.modalities,
                "applicability_note": (
                    "Hounsfield-unit preset for CT."
                    if mod in preset.modalities
                    else f"Defined in Hounsfield units; not directly applicable to {mod or 'this modality'}."
                ),
            }
        )
    return out


def apply_window(array: np.ndarray, center: float, width: float) -> np.ndarray:
    """Map intensities to 0..255 using a linear window/level.

    ``width`` is the full window width (DICOM ``WindowWidth`` semantics).
    """
    width = max(float(width), 1e-6)
    low = float(center) - width / 2.0
    high = float(center) + width / 2.0
    scaled = (np.asarray(array, dtype=np.float32) - low) / (high - low)
    return np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)


def auto_window(array: np.ndarray, *, low_percentile: float = 0.5, high_percentile: float = 99.5) -> tuple[float, float]:
    """Robust window/level from percentiles (used for non-CT or missing tags)."""
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return (0.0, 1.0)
    low = float(np.percentile(finite, low_percentile))
    high = float(np.percentile(finite, high_percentile))
    if high - low < 1e-6:
        high = low + 1.0
    return ((low + high) / 2.0, high - low)


def _normalize(array: np.ndarray, mode: str) -> tuple[np.ndarray, list[str]]:
    notes: list[str] = []
    if mode == "none":
        return array, notes
    if mode == "minmax":
        lo = float(np.min(array))
        hi = float(np.max(array))
        span = hi - lo
        if span < 1e-9:
            notes.append("Min/max normalisation skipped: the volume has a single intensity value.")
            return array.astype(np.float32), notes
        notes.append(f"Min-max normalised to 0..1 (source range {lo:.3g}..{hi:.3g}).")
        return ((array.astype(np.float32) - lo) / span), notes
    if mode == "zscore":
        mean = float(np.mean(array))
        std = float(np.std(array))
        if std < 1e-9:
            notes.append("Z-score normalisation skipped: zero intensity variance.")
            return array.astype(np.float32), notes
        notes.append(f"Z-score normalised (mean {mean:.3g}, sd {std:.3g}).")
        return ((array.astype(np.float32) - mean) / std), notes
    raise PreprocessingError(
        f"Unknown normalisation mode '{mode}'. Supported: none, minmax, zscore."
    )


def _denoise(array: np.ndarray, options: PreprocessingOptions) -> tuple[np.ndarray, list[str]]:
    notes: list[str] = []
    if options.denoise == "none":
        return array, notes
    if options.denoise == "gaussian":
        sigma = max(float(options.denoise_sigma), 0.0)
        if sigma <= 0:
            return array, notes
        notes.append(f"Gaussian smoothing applied (sigma = {sigma:g} voxels, 3D).")
        return ndimage.gaussian_filter(array.astype(np.float32), sigma=sigma), notes
    if options.denoise == "median":
        size = max(1, int(options.denoise_size))
        if size % 2 == 0:
            size += 1
        if size == 1:
            return array, notes
        notes.append(f"Median filter applied ({size}x{size}x{size} voxels).")
        return ndimage.median_filter(array.astype(np.float32), size=size), notes
    raise PreprocessingError(
        f"Unknown denoise mode '{options.denoise}'. Supported: none, gaussian, median."
    )


def preprocess_volume(volume: Volume, options: PreprocessingOptions, *, progress=None) -> Volume:
    """Apply the requested preprocessing chain and return a new volume."""
    array = np.asarray(volume.array)
    spacing = volume.spacing
    notes: list[str] = list(volume.notes)
    warnings: list[str] = list(volume.warnings)
    normalized = volume.is_normalized
    units = volume.intensity_units
    resampled = volume.is_resampled

    if options.clip_min is not None or options.clip_max is not None:
        lo = -np.inf if options.clip_min is None else float(options.clip_min)
        hi = np.inf if options.clip_max is None else float(options.clip_max)
        if not lo < hi:
            raise PreprocessingError("Intensity clip range is empty (minimum is not below maximum).")
        clipped = np.clip(array, lo, hi)
        changed = int(np.count_nonzero(clipped != array))
        notes.append(f"Intensity clipping to [{lo:g}, {hi:g}] affected {changed} voxel(s).")
        array = clipped
        if progress:
            progress.step("preprocessing", 0.75, f"Clipped intensities to [{lo:g}, {hi:g}]")

    array, denoise_notes = _denoise(array, options)
    notes.extend(denoise_notes)
    if denoise_notes and progress:
        progress.step("preprocessing", 0.78, denoise_notes[-1])

    if options.normalize != "none":
        array, norm_notes = _normalize(np.asarray(array, dtype=np.float32), options.normalize)
        notes.extend(norm_notes)
        normalized = True
        units = "normalized"
        warnings.append(
            "The volume was intensity-normalised, so Hounsfield-unit threshold presets are no "
            "longer meaningful. Use manual thresholds for segmentation."
        )
        if progress:
            progress.step("preprocessing", 0.80, norm_notes[-1] if norm_notes else "Normalised")

    if options.normalize == "none" and not np.issubdtype(np.asarray(array).dtype, np.floating):
        array = np.asarray(array)

    target = options.resample_target_mm
    if target is None and options.isotropic:
        base = np.array(spacing, dtype=np.float64)
        value = float(np.min(base[base > 0])) if np.any(base > 0) else 1.0
        target = (value, value, value)
    if target is not None:
        target = (float(target[0]), float(target[1]), float(target[2]))
        if any(t <= 0 for t in target):
            raise PreprocessingError("Resample target spacing must be strictly positive.")
        if any(abs(spacing[a] - target[a]) > 1e-4 for a in range(3)):
            if progress:
                progress.step(
                    "preprocessing",
                    0.82,
                    "Resampling to {:.3f} x {:.3f} x {:.3f} mm".format(*target),
                )
            array, new_spacing = resample_volume(np.asarray(array), spacing, target)
            notes.append(
                "Resampled to {:.3f} x {:.3f} x {:.3f} mm.".format(*new_spacing)
            )
            spacing = new_spacing
            resampled = True

    factor = int(options.downsample_factor or 1)
    if factor > 1:
        array = np.asarray(array)[::factor, ::factor, ::factor]
        spacing = (spacing[0] * factor, spacing[1] * factor, spacing[2] * factor)
        notes.append(f"Downsampled by a factor of {factor}; effective spacing {spacing} mm.")
        warnings.append(
            "The volume was downsampled. Physical dimensions are still correct but fine anatomical "
            "detail is reduced."
        )
        resampled = True

    return volume.copy_with(
        array=array,
        spacing=spacing,
        is_normalized=normalized,
        is_resampled=resampled,
        intensity_units=units,
        notes=notes,
        warnings=warnings,
    )


def intensity_statistics(array: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """Descriptive statistics, optionally restricted to a mask."""
    data = np.asarray(array, dtype=np.float32)
    if mask is not None and np.any(mask):
        values = data[mask]
    else:
        values = data.reshape(-1)
    if values.size == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p2": float(np.percentile(values, 2)),
        "p98": float(np.percentile(values, 98)),
    }
