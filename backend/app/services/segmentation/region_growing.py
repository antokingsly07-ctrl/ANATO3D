"""Connected-threshold (region growing) segmentation.

Implementation notes
--------------------
Region growing is implemented as a *geodesic reconstruction*: an acceptance mask
is built from the intensity criteria, a marker is placed at the seed voxel(s),
and morphological reconstruction by dilation extracts exactly the connected
component(s) of the acceptance mask that contain the seeds.  This is the
classical "connected threshold" algorithm and runs in C (scikit-image), which
keeps it usable on full-resolution volumes.

Two intensity criteria are supported:

``window``     accept voxels inside ``[seed_value - tolerance, seed_value + tolerance]``
``absolute``   accept voxels inside ``[threshold_min, threshold_max]``
"""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.morphology import reconstruction

from ...core.errors import SegmentationError
from ...core.logging_ import get_logger
from ...models import SegmentationMask, Volume
from .base import Availability, ParameterSpec, SegmentationEngine, cleanup_kwargs, validate_mask_non_empty
from .refine import count_components, refine_mask

logger = get_logger("segmentation.region_growing")


class RegionGrowingSegmentation(SegmentationEngine):
    key = "region_growing"
    display_name = "Region growing"
    description = (
        "Grows a region from one or more seed voxels through connected voxels that satisfy an "
        "intensity criterion. Useful for homogeneous structures such as cortical bone or a "
        "fluid-filled space."
    )
    category = "classical"
    requires_seeds = True
    uses_hounsfield = True

    def availability(self, volume: Volume) -> Availability:
        return Availability(True)

    def parameters(self) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                "seed_mode", "Seed selection", "choice", "auto_largest",
                choices=("manual", "auto_largest", "auto_brightest"),
                help=(
                    "manual = seeds supplied as voxel indices; auto_largest = centroid of the largest "
                    "component of the threshold mask; auto_brightest = the most intense voxels."
                ),
            ),
            ParameterSpec("seed_points", "Seed voxel indices (k, j, i)", "text", [],
                          help="Only used when seed selection is 'manual'.", advanced=True),
            ParameterSpec("seed_count", "Number of automatic seeds", "int", 1, 1, 16, advanced=True),
            ParameterSpec(
                "criterion", "Intensity criterion", "choice", "window",
                choices=("window", "absolute"),
                help="'window' grows around the seed intensity by the tolerance; 'absolute' uses a fixed range.",
            ),
            ParameterSpec("tolerance", "Tolerance", "float", 120.0, 1.0, 4000.0, unit="HU/au",
                          help="Half-width of the acceptance window around each seed value."),
            ParameterSpec("threshold_min", "Absolute minimum", "float", 150.0, unit="HU/au", advanced=True),
            ParameterSpec("threshold_max", "Absolute maximum", "float", 3071.0, unit="HU/au", advanced=True),
            *super().parameters(),
        ]

    # -- seed handling -----------------------------------------------------
    def _resolve_seeds(self, volume: Volume, params: dict[str, Any], acceptance: np.ndarray) -> tuple[list[tuple[int, int, int]], str]:
        mode = str(params.get("seed_mode", "auto_largest"))
        array = np.asarray(volume.array)
        shape = array.shape

        if mode == "manual":
            raw = params.get("seed_points") or []
            if isinstance(raw, str):
                raw = [
                    [int(v) for v in part.strip().strip("[]()").split(",")]
                    for part in raw.split(";")
                    if part.strip()
                ]
            seeds: list[tuple[int, int, int]] = []
            for item in raw:
                try:
                    k, j, i = (int(round(float(v))) for v in item)
                except (TypeError, ValueError) as exc:
                    raise SegmentationError(
                        "Seed points must be voxel indices in (slice, row, column) order."
                    ) from exc
                if not (0 <= k < shape[0] and 0 <= j < shape[1] and 0 <= i < shape[2]):
                    raise SegmentationError(
                        f"Seed point {k}, {j}, {i} lies outside the volume of shape {shape}."
                    )
                seeds.append((k, j, i))
            if not seeds:
                raise SegmentationError(
                    "Manual seed selection was requested but no seed points were supplied."
                )
            return seeds, f"{len(seeds)} manual seed(s)."

        count = max(1, int(params.get("seed_count", 1) or 1))

        if mode == "auto_largest":
            if not np.any(acceptance):
                raise SegmentationError(
                    "Region growing could not place a seed: the intensity criterion matches no voxels.",
                    detail="Widen the tolerance or the absolute threshold range.",
                )
            work = acceptance
            seeds = []
            for _ in range(count):
                if not np.any(work):
                    break
                coords = np.argwhere(work)
                # largest component of the current acceptance mask
                from scipy import ndimage

                labels, n = ndimage.label(work, structure=ndimage.generate_binary_structure(3, 3))
                if n == 0:
                    break
                sizes = np.bincount(labels.ravel(), minlength=n + 1)
                sizes[0] = 0
                biggest = int(np.argmax(sizes))
                component = labels == biggest
                centroid = np.argwhere(component).mean(axis=0)
                seed = tuple(int(round(v)) for v in np.clip(centroid, 0, np.array(shape) - 1))
                seeds.append(seed)  # type: ignore[arg-type]
                work = work & ~component
            return seeds, f"{len(seeds)} automatic seed(s) from the largest accepted component(s)."

        if mode == "auto_brightest":
            flat = np.argsort(array.reshape(-1))[::-1][: max(count, count * 50)]
            seeds = []
            for flat_index in flat:
                seed = tuple(int(v) for v in np.unravel_index(int(flat_index), shape))
                if acceptance[seed]:
                    seeds.append(seed)  # type: ignore[arg-type]
                if len(seeds) >= count:
                    break
            if not seeds:
                raise SegmentationError(
                    "Region growing could not place a seed: the brightest voxels fall outside the "
                    "intensity criterion."
                )
            return seeds, f"{len(seeds)} automatic seed(s) at the brightest accepted voxels."

        raise SegmentationError(
            f"Unknown seed mode '{mode}'. Use 'manual', 'auto_largest' or 'auto_brightest'."
        )

    # -- run ---------------------------------------------------------------
    def run(self, volume: Volume, params: dict[str, Any], *, progress=None) -> SegmentationMask:
        array = np.asarray(volume.array)
        criterion = str(params.get("criterion", "window"))
        tolerance = float(params.get("tolerance", 120.0))
        notes: list[str] = []
        warnings: list[str] = []

        if criterion == "window":
            # Provisional acceptance used only for seeding; refined per seed below.
            lo = float(np.min(array))
            hi = float(np.max(array))
            acceptance = np.ones(array.shape, dtype=bool)
            if progress:
                progress.step("segmentation", 0.06, "Region growing: resolving seeds")
            seeds, seed_note = self._resolve_seeds(volume, params, acceptance)
            notes.append(f"Intensity criterion: seed value +/- {tolerance:g} {volume.intensity_units}.")
        elif criterion == "absolute":
            lo = float(params.get("threshold_min", -1100.0))
            hi = float(params.get("threshold_max", 3071.0))
            if not lo < hi:
                raise SegmentationError("Absolute criterion is empty: minimum must be below maximum.")
            acceptance = (array >= lo) & (array <= hi)
            if progress:
                progress.step("segmentation", 0.06, "Region growing: resolving seeds")
            seeds, seed_note = self._resolve_seeds(volume, params, acceptance)
            notes.append(f"Intensity criterion: absolute range {lo:g} .. {hi:g} {volume.intensity_units}.")
        else:
            raise SegmentationError(f"Unknown criterion '{criterion}'.")

        notes.append(seed_note)
        notes.append("Seeds: " + ", ".join(f"({k}, {j}, {i})" for k, j, i in seeds[:8]) + ("..." if len(seeds) > 8 else ""))

        marker = np.zeros(array.shape, dtype=bool)
        for k, j, i in seeds:
            marker[k, j, i] = True

        if criterion == "window":
            combined = np.zeros(array.shape, dtype=bool)
            for k, j, i in seeds:
                value = float(array[k, j, i])
                window = (array >= value - tolerance) & (array <= value + tolerance)
                component = reconstruction(marker, window, method="dilation")
                combined |= component
                if progress:
                    progress.step(
                        "segmentation",
                        0.10,
                        f"Grew region from seed ({k}, {j}, {i}) at {value:.0f} {volume.intensity_units}",
                    )
            mask = combined
        else:
            mask = reconstruction(marker, acceptance, method="dilation")

        before = int(np.count_nonzero(mask))
        if before == 0:
            raise SegmentationError(
                "Region growing produced an empty mask: the seeds are not connected to any voxel "
                "that satisfies the intensity criterion.",
                detail="Increase the tolerance or move the seed into the structure of interest.",
            )
        if progress:
            progress.step("segmentation", 0.28, f"Region growing selected {before} voxel(s)")

        refined, report = refine_mask(
            mask,
            voxel_volume_mm3=volume.voxel_volume_mm3,
            **cleanup_kwargs(params),
            progress=progress,
        )
        notes.append(
            "Refinement: " + (" ".join(report.steps) if report.steps else "no operations applied.")
        )
        if report.output_components > 1:
            warnings.append(
                f"The final mask contains {report.output_components} disconnected components; "
                "consider enabling 'keep largest N components' if only one structure is expected."
            )

        validate_mask_non_empty(
            refined,
            engine=self.display_name,
            hint=(
                "Mesh generation failed because no foreground voxels were detected. The grown region "
                "was entirely removed by the refinement steps; reduce the minimum component size."
            ),
        )

        return SegmentationMask(
            mask=refined,
            label=str(params.get("label") or "Grown region"),
            color=str(params.get("color") or "#4fc3f7"),
            engine=self.key,
            params={**params, "resolved_seeds": seeds, "resolved_criterion": criterion},
            notes=notes,
            warnings=warnings,
        )
