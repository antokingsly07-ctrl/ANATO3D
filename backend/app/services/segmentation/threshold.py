"""Threshold-based segmentation.

Two selection modes:

``absolute``
    Intensity window ``[threshold_min, threshold_max]``.  For CT the values are
    Hounsfield units, so a window such as 200..3000 HU isolates bone.

``percentile``
    The window bounds are derived from percentiles of the *current volume*, which
    is the modality-independent option for MRI where absolute units are not
    defined.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core.errors import SegmentationError
from ...core.logging_ import get_logger
from ...models import SegmentationMask, Volume
from .base import Availability, ParameterSpec, SegmentationEngine, cleanup_kwargs, validate_mask_non_empty
from .refine import mask_statistics, refine_mask

logger = get_logger("segmentation.threshold")


class ThresholdSegmentation(SegmentationEngine):
    key = "threshold"
    display_name = "Threshold"
    description = (
        "Selects voxels whose intensity falls inside a range. For CT the range is in Hounsfield "
        "units, which makes it directly comparable with clinical windowing."
    )
    category = "classical"
    uses_hounsfield = True

    def availability(self, volume: Volume) -> Availability:
        return Availability(True)

    def parameters(self) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                "mode", "Selection mode", "choice", "absolute",
                choices=("absolute", "percentile"),
                help="Absolute intensity window, or percentile-based selection (modality independent).",
            ),
            ParameterSpec("threshold_min", "Minimum", "float", 200.0, unit="HU/au"),
            ParameterSpec("threshold_max", "Maximum", "float", 3071.0, unit="HU/au"),
            ParameterSpec("lower_percentile", "Lower percentile", "float", 97.0, 0.0, 100.0,
                          step=0.5, unit="%", advanced=True),
            ParameterSpec("upper_percentile", "Upper percentile", "float", 100.0, 0.0, 100.0,
                          step=0.5, unit="%", advanced=True),
            ParameterSpec("invert", "Invert selection", "bool", False, advanced=True),
            *super().parameters(),
        ]

    def run(self, volume: Volume, params: dict[str, Any], *, progress=None) -> SegmentationMask:
        array = np.asarray(volume.array)
        mode = str(params.get("mode", "absolute"))

        if mode == "percentile":
            lower = float(params.get("lower_percentile", 97.0))
            upper = float(params.get("upper_percentile", 100.0))
            if not 0.0 <= lower < upper <= 100.0:
                raise SegmentationError(
                    "Percentile range is invalid: the lower percentile must be below the upper one "
                    "and both must lie between 0 and 100."
                )
            lo = float(np.percentile(array, lower))
            hi = float(np.percentile(array, upper))
            notes_mode = f"Percentile selection: p{lower:g} = {lo:.3g}, p{upper:g} = {hi:.3g}."
        elif mode == "absolute":
            lo = float(params.get("threshold_min", 0.0))
            hi = float(params.get("threshold_max", 1.0))
            if not lo < hi:
                raise SegmentationError(
                    "Threshold range is empty: the minimum must be strictly below the maximum."
                )
            if volume.intensity_units == "HU":
                if lo < -1100 or hi > 4000:
                    logger.debug("Threshold window %s..%s HU is outside the typical CT range", lo, hi)
            notes_mode = f"Absolute window: {lo:g} .. {hi:g} {volume.intensity_units}."
        else:
            raise SegmentationError(f"Unknown threshold mode '{mode}'. Use 'absolute' or 'percentile'.")

        if progress:
            progress.step("segmentation", 0.06, "Applying threshold " + notes_mode)

        mask = (array >= lo) & (array <= hi)
        if bool(params.get("invert", False)):
            mask = ~mask
            notes_mode += " Selection inverted."

        notes = [notes_mode]
        warnings: list[str] = []
        if volume.is_normalized and volume.intensity_units != "HU" and mode == "absolute":
            warnings.append(
                "The volume is intensity-normalised, so absolute thresholds no longer correspond to "
                "Hounsfield units. Percentile mode is recommended in this situation."
            )

        if progress:
            progress.step(
                "segmentation",
                0.15,
                f"Threshold selected {int(np.count_nonzero(mask))} voxel(s) before refinement",
            )

        refined, report = refine_mask(
            mask,
            voxel_volume_mm3=volume.voxel_volume_mm3,
            **cleanup_kwargs(params),
            progress=progress,
        )
        if progress:
            progress.step(
                "segmentation",
                0.35,
                f"Refinement produced {int(np.count_nonzero(refined))} voxel(s) in "
                f"{report.output_components} component(s)",
            )

        notes.append(f"Refinement: {' '.join(report.steps) if report.steps else 'no operations applied.'}")
        validate_mask_non_empty(
            refined,
            engine=self.display_name,
            hint=(
                "Mesh generation failed because no foreground voxels were detected. "
                f"The window {lo:g}..{hi:g} {volume.intensity_units} contained no voxels that survived "
                "refinement; widen the range or reduce the minimum component size."
            ),
        )

        return SegmentationMask(
            mask=refined,
            label=str(params.get("label") or "Threshold region"),
            color=str(params.get("color") or "#e8b04b"),
            engine=self.key,
            params={**params, "resolved_min": lo, "resolved_max": hi},
            notes=notes,
            warnings=warnings,
        )
