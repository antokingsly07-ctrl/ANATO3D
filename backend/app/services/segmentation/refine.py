"""Mask refinement primitives shared by all segmentation engines.

Every operation is explicit about what it changed so the UI can show a real
refinement report instead of a black box.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage.morphology import ball

from ...core.errors import SegmentationError
from ...core.logging_ import get_logger

logger = get_logger("segmentation.refine")

CONNECTIVITY_STRUCTURES = {
    1: ndimage.generate_binary_structure(3, 1),  # 6-neighbourhood
    2: ndimage.generate_binary_structure(3, 2),  # 18-neighbourhood
    3: ndimage.generate_binary_structure(3, 3),  # 26-neighbourhood
}


@dataclass
class RefinementReport:
    input_voxels: int = 0
    output_voxels: int = 0
    input_components: int = 0
    output_components: int = 0
    removed_components: int = 0
    removed_voxels: int = 0
    holes_filled_voxels: int = 0
    steps: list[str] = field(default_factory=list)

    @property
    def volume_change_pct(self) -> float:
        if not self.input_voxels:
            return 0.0
        return 100.0 * (self.output_voxels - self.input_voxels) / self.input_voxels

    def as_dict(self) -> dict[str, object]:
        return {
            "input_voxels": self.input_voxels,
            "output_voxels": self.output_voxels,
            "input_components": self.input_components,
            "output_components": self.output_components,
            "removed_components": self.removed_components,
            "removed_voxels": self.removed_voxels,
            "holes_filled_voxels": self.holes_filled_voxels,
            "volume_change_pct": round(self.volume_change_pct, 3),
            "steps": list(self.steps),
        }


def count_components(mask: np.ndarray, connectivity: int = 3) -> int:
    if not np.any(mask):
        return 0
    _, count = ndimage.label(mask, structure=CONNECTIVITY_STRUCTURES[connectivity])
    return int(count)


def component_sizes(mask: np.ndarray, connectivity: int = 3) -> np.ndarray:
    """Voxel counts of every connected component, largest first."""
    if not np.any(mask):
        return np.zeros(0, dtype=np.int64)
    labels, count = ndimage.label(mask, structure=CONNECTIVITY_STRUCTURES[connectivity])
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    return np.sort(sizes)[::-1]


def fill_holes(mask: np.ndarray, *, connectivity: int = 1) -> tuple[np.ndarray, int]:
    """Fill enclosed cavities (3D flood fill from the volume border)."""
    if not np.any(mask):
        return mask, 0
    structure = CONNECTIVITY_STRUCTURES[connectivity]
    background = ~mask
    labels, _ = ndimage.label(background, structure=structure)
    border_labels = set()
    for plane in (
        labels[0, :, :], labels[-1, :, :],
        labels[:, 0, :], labels[:, -1, :],
        labels[:, :, 0], labels[:, :, -1],
    ):
        border_labels.update(np.unique(plane).tolist())
    border_labels.discard(0)
    if not border_labels:
        return mask, 0
    outside = np.isin(labels, list(border_labels))
    holes = background & ~outside
    filled = mask | holes
    return filled, int(np.count_nonzero(holes))


def _binary_op(mask: np.ndarray, radius: int, mode: str) -> np.ndarray:
    if radius <= 0:
        return mask
    structure = ball(int(radius))
    if mode == "close":
        return ndimage.binary_closing(mask, structure=structure, border_value=0)
    if mode == "open":
        return ndimage.binary_opening(mask, structure=structure, border_value=0)
    if mode == "dilate":
        return ndimage.binary_dilation(mask, structure=structure)
    if mode == "erode":
        return ndimage.binary_erosion(mask, structure=structure, border_value=0)
    raise SegmentationError(f"Unknown morphology operation '{mode}'.")


def remove_small_components(
    mask: np.ndarray,
    *,
    min_voxels: int = 0,
    keep_largest_n: int = 0,
    connectivity: int = 3,
) -> tuple[np.ndarray, int, int]:
    """Drop small / spurious components. Returns (mask, removed_components, removed_voxels)."""
    if not np.any(mask):
        return mask, 0, 0
    labels, count = ndimage.label(mask, structure=CONNECTIVITY_STRUCTURES[connectivity])
    if count == 0:
        return mask, 0, 0
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0

    keep = np.ones(count + 1, dtype=bool)
    keep[0] = False
    removed_components = 0

    if min_voxels > 0:
        small = np.where(sizes[1:] < int(min_voxels))[0] + 1
        keep[small] = False
        removed_components += int(small.size)

    if keep_largest_n > 0:
        order = np.argsort(sizes[1:])[::-1] + 1
        allowed = set(order[: int(keep_largest_n)].tolist())
        for label_id in np.where(keep)[0]:
            if label_id != 0 and label_id not in allowed:
                keep[label_id] = False

    kept_labels = np.where(keep)[0]
    removed_components = int(count - len(kept_labels))
    if removed_components == 0:
        return mask, 0, 0
    result = keep[labels]
    removed_voxels = int(np.count_nonzero(mask) - np.count_nonzero(result))
    return result, removed_components, removed_voxels


def smooth_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth a binary mask by Gaussian blurring and re-thresholding at 0.5."""
    if sigma <= 0:
        return mask
    blurred = ndimage.gaussian_filter(mask.astype(np.float32), sigma=float(sigma))
    return blurred >= 0.5


def refine_mask(
    mask: np.ndarray,
    *,
    voxel_volume_mm3: float,
    fill_holes_flag: bool = True,
    close_radius: int = 0,
    open_radius: int = 0,
    dilate_radius: int = 0,
    erode_radius: int = 0,
    smooth_sigma: float = 0.0,
    min_component_mm3: float = 0.0,
    keep_largest_n: int = 0,
    connectivity: int = 3,
    progress=None,
) -> tuple[np.ndarray, RefinementReport]:
    """Apply the standard refinement chain to a binary mask."""
    report = RefinementReport()
    work = np.asarray(mask, dtype=bool)
    report.input_voxels = int(np.count_nonzero(work))
    report.input_components = count_components(work, connectivity)

    if not np.any(work):
        report.output_voxels = 0
        return work, report

    if fill_holes_flag:
        work, filled = fill_holes(work)
        if filled:
            report.holes_filled_voxels = filled
            report.steps.append(f"Filled {filled} enclosed voxel(s).")

    if close_radius > 0:
        before = int(np.count_nonzero(work))
        work = _binary_op(work, close_radius, "close")
        report.steps.append(
            f"Morphological closing (radius {close_radius}) added "
            f"{int(np.count_nonzero(work)) - before} voxel(s)."
        )
        if progress:
            progress.log(report.steps[-1], stage="segmentation")

    if open_radius > 0:
        before = int(np.count_nonzero(work))
        work = _binary_op(work, open_radius, "open")
        report.steps.append(
            f"Morphological opening (radius {open_radius}) removed {before - int(np.count_nonzero(work))} voxel(s)."
        )

    if dilate_radius > 0:
        work = _binary_op(work, dilate_radius, "dilate")
        report.steps.append(f"Dilated by {dilate_radius} voxel(s).")

    if erode_radius > 0:
        work = _binary_op(work, erode_radius, "erode")
        report.steps.append(f"Eroded by {erode_radius} voxel(s).")

    if smooth_sigma > 0:
        work = smooth_mask(work, smooth_sigma)
        report.steps.append(f"Mask smoothed with a Gaussian surface (sigma {smooth_sigma:g} voxels).")

    min_voxels = 0
    if min_component_mm3 > 0:
        if voxel_volume_mm3 <= 0:
            raise SegmentationError("Voxel volume is invalid; cannot apply a physical size filter.")
        min_voxels = int(round(float(min_component_mm3) / voxel_volume_mm3))
        report.steps.append(
            f"Discarding components smaller than {min_component_mm3:g} mm3 ({min_voxels} voxels)."
        )

    if min_voxels > 0 or keep_largest_n > 0:
        work, removed_components, removed_voxels = remove_small_components(
            work,
            min_voxels=min_voxels,
            keep_largest_n=keep_largest_n,
            connectivity=connectivity,
        )
        report.removed_components = removed_components
        report.removed_voxels = removed_voxels
        if removed_components:
            report.steps.append(
                f"Removed {removed_components} disconnected component(s) ({removed_voxels} voxels)."
            )

    report.output_voxels = int(np.count_nonzero(work))
    report.output_components = count_components(work, connectivity)
    return work, report


def mask_statistics(mask: np.ndarray, voxel_volume_mm3: float) -> dict[str, object]:
    voxels = int(np.count_nonzero(mask))
    return {
        "voxel_count": voxels,
        "volume_mm3": round(voxels * voxel_volume_mm3, 3),
        "volume_cm3": round(voxels * voxel_volume_mm3 / 1000.0, 4),
        "components": count_components(mask, 3),
        "bounding_box_index": _bounding_box(mask),
    }


def _bounding_box(mask: np.ndarray) -> list[list[int]] | None:
    if not np.any(mask):
        return None
    coords = np.argwhere(mask)
    lower = coords.min(axis=0).tolist()
    upper = (coords.max(axis=0) + 1).tolist()
    return [lower, upper]
