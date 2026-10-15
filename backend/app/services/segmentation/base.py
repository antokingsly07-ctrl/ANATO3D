"""Segmentation engine abstraction.

The pipeline is intentionally engine-agnostic: every algorithm implements
:class:`SegmentationEngine` and returns a :class:`SegmentationMask`.  Adding a
deep-learning model later (MONAI / nnU-Net / a custom TorchScript export) means
implementing two methods -- ``availability`` and ``run`` -- and registering the
class; nothing else in the application has to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...models import SegmentationMask, Volume


@dataclass(frozen=True)
class ParameterSpec:
    """Declarative description of one engine parameter (drives the UI form)."""

    key: str
    label: str
    kind: str  # "float" | "int" | "bool" | "choice"
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] | None = None
    unit: str | None = None
    help: str = ""
    advanced: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "choices": list(self.choices) if self.choices else None,
            "unit": self.unit,
            "help": self.help,
            "advanced": self.advanced,
        }


@dataclass
class Availability:
    available: bool
    reason: str | None = None
    requirements: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "requirements": list(self.requirements),
        }


class SegmentationEngine(ABC):
    """Base class for all segmentation algorithms."""

    #: stable machine key, e.g. ``threshold``
    key: str = "base"
    display_name: str = "Base engine"
    description: str = ""
    #: "classical" | "ai"
    category: str = "classical"
    #: parameter keys that must be provided by the caller
    requires_seeds: bool = False
    #: whether the engine interprets values in Hounsfield units
    uses_hounsfield: bool = True

    @abstractmethod
    def availability(self, volume: Volume) -> Availability:
        """Report whether this engine can run on ``volume`` right now."""

    @abstractmethod
    def run(
        self,
        volume: Volume,
        params: dict[str, Any],
        *,
        progress=None,
    ) -> SegmentationMask:
        """Produce a mask.  Must raise :class:`SegmentationError` subclasses on failure."""

    def parameters(self) -> list[ParameterSpec]:
        return []

    def describe(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category,
            "requires_seeds": self.requires_seeds,
            "uses_hounsfield": self.uses_hounsfield,
            "parameters": [spec.as_dict() for spec in self.parameters()],
        }


# ---------------------------------------------------------------------------
# Shared parameter blocks
# ---------------------------------------------------------------------------

def cleanup_parameters() -> list[ParameterSpec]:
    return [
        ParameterSpec(
            "fill_holes", "Fill enclosed holes", "bool", True,
            help="Fills cavities fully enclosed by the mask (e.g. marrow inside cortical bone).",
        ),
        ParameterSpec("close_radius", "Closing radius", "int", 1, 0, 10,
                      unit="voxels", help="Bridges small gaps and connects nearby fragments."),
        ParameterSpec("open_radius", "Opening radius", "int", 0, 0, 10,
                      unit="voxels", help="Removes thin bridges and speckle."),
        ParameterSpec("min_component_mm3", "Minimum component size", "float", 50.0, 0.0, 100000.0,
                      step=10.0, unit="mm3", help="Components smaller than this are discarded.", advanced=True),
        ParameterSpec("keep_largest_n", "Keep largest N components", "int", 0, 0, 50,
                      unit="components", help="0 keeps all components that pass the size filter.", advanced=True),
        ParameterSpec("connectivity", "Component connectivity", "choice", 3,
                      choices=("1", "2", "3"), help="1 = faces, 2 = faces+edges, 3 = faces+edges+corners.", advanced=True),
    ]


def cleanup_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "fill_holes_flag": bool(params.get("fill_holes", True)),
        "close_radius": int(params.get("close_radius", 0) or 0),
        "open_radius": int(params.get("open_radius", 0) or 0),
        "dilate_radius": int(params.get("dilate_radius", 0) or 0),
        "erode_radius": int(params.get("erode_radius", 0) or 0),
        "smooth_sigma": float(params.get("smooth_sigma", 0.0) or 0.0),
        "min_component_mm3": float(params.get("min_component_mm3", 0.0) or 0.0),
        "keep_largest_n": int(params.get("keep_largest_n", 0) or 0),
        "connectivity": int(params.get("connectivity", 3) or 3),
    }


def validate_mask_non_empty(mask: np.ndarray, *, engine: str, hint: str = "") -> None:
    from ...core.errors import EmptyMaskError

    if not np.any(mask):
        raise EmptyMaskError(
            f"Segmentation with the '{engine}' engine produced an empty mask.",
            detail=(
                hint
                or "Mesh generation failed because no foreground voxels were detected. "
                "Widen the threshold range or disable aggressive morphological opening."
            ),
        )
