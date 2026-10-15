"""Session-scoped wrapper records held in the in-memory :class:`AppStore`.

The store keeps every artefact produced during a session.  Raw domain objects
(:class:`Volume`, :class:`SegmentationResult`, ...) are wrapped so that each
record carries the ``dataset_id`` it belongs to -- that is what the background
reaper uses to evict everything derived from an expired workspace.

These wrappers deliberately hold *no* HTTP knowledge; the API layer converts
them into Pydantic response models (see ``app/api/presenters.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .domain import DatasetRef, MeshAsset, SegmentationResult, Volume

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from ..services.dicom.series import DatasetAnalysis
    from ..services.validation.ground_truth import GroundTruth


@dataclass
class DatasetRecord:
    """An uploaded (or synthetic) dataset plus its cached analysis."""

    dataset_id: str
    dataset: DatasetRef
    root: Path
    is_synthetic: bool = False
    synthetic_notes: list[str] = field(default_factory=list)
    non_image_files: int = 0
    unreadable_files: int = 0
    warnings: list[str] = field(default_factory=list)
    analysis: "DatasetAnalysis | None" = None
    ground_truth_id: str | None = None


@dataclass
class VolumeRecord:
    """A reconstructed volume with its report and validation findings."""

    dataset_id: str
    volume: Volume
    ordering: Any = None
    report: dict[str, Any] = field(default_factory=dict)
    findings: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    series_instance_uid: str | None = None

    @property
    def volume_id(self) -> str:
        return self.volume.volume_id


@dataclass
class SegmentationRecord:
    dataset_id: str
    result: SegmentationResult
    findings: list[dict[str, Any]] = field(default_factory=list)

    @property
    def segmentation_id(self) -> str:
        return self.result.segmentation_id


@dataclass
class MeshRecord:
    dataset_id: str
    asset: MeshAsset

    @property
    def mesh_id(self) -> str:
        return self.asset.mesh_id


@dataclass
class ValidationRecord:
    dataset_id: str | None
    report: dict[str, Any]

    @property
    def validation_id(self) -> str:
        return str(self.report.get("validation_id", ""))


@dataclass
class AnalysisRecord:
    dataset_id: str | None
    payload: dict[str, Any]


@dataclass
class GroundTruthRecord:
    dataset_id: str
    ground_truth_id: str
    filename: str
    ground_truth: "GroundTruth"


__all__ = [
    "AnalysisRecord",
    "DatasetRecord",
    "GroundTruthRecord",
    "MeshRecord",
    "SegmentationRecord",
    "ValidationRecord",
    "VolumeRecord",
]
