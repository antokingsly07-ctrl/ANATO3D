"""Engine registry and anatomical presets."""

from __future__ import annotations

from typing import Any

from ...core.errors import ValidationError
from ...models import Volume
from .ai import AISegmentation, ClassicalFallbackSegmentation
from .base import SegmentationEngine
from .region_growing import RegionGrowingSegmentation
from .threshold import ThresholdSegmentation

#: All registered engines, in display order.
ENGINES: tuple[SegmentationEngine, ...] = (
    ThresholdSegmentation(),
    RegionGrowingSegmentation(),
    AISegmentation(),
    ClassicalFallbackSegmentation(),
)

ENGINE_MAP: dict[str, SegmentationEngine] = {engine.key: engine for engine in ENGINES}


def get_engine(key: str) -> SegmentationEngine:
    engine = ENGINE_MAP.get(key)
    if engine is None:
        raise ValidationError(
            f"Unknown segmentation engine '{key}'. Available engines: "
            + ", ".join(sorted(ENGINE_MAP)),
        )
    return engine


def list_engines(volume: Volume | None = None) -> list[dict[str, Any]]:
    out = []
    for engine in ENGINES:
        payload = engine.describe()
        payload["availability"] = (
            engine.availability(volume).as_dict() if volume is not None else {"available": True, "reason": None}
        )
        out.append(payload)
    return out


# ---------------------------------------------------------------------------
# Anatomical presets
# ---------------------------------------------------------------------------

def _preset(
    key: str,
    label: str,
    engine: str,
    modality_applicability: list[str],
    params: dict[str, Any],
    description: str,
    color: str,
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "engine": engine,
        "modalities": modality_applicability,
        "params": params,
        "description": description,
        "color": color,
    }


PRESETS: tuple[dict[str, Any], ...] = (
    _preset(
        "ct_bone",
        "Bone (cortical + trabecular)",
        "threshold",
        ["CT", "CBCT"],
        {
            "mode": "absolute",
            "threshold_min": 200.0,
            "threshold_max": 3071.0,
            "fill_holes": True,
            "close_radius": 1,
            "open_radius": 1,
            "min_component_mm3": 50.0,
            "keep_largest_n": 0,
            "label": "Bone",
            "color": "#e8d5a8",
        },
        "Selects voxels above +200 HU, which captures cortical bone and dense calcification while "
        "excluding soft tissue.",
        "#e8d5a8",
    ),
    _preset(
        "ct_dense_bone",
        "Dense bone only",
        "threshold",
        ["CT", "CBCT"],
        {
            "mode": "absolute",
            "threshold_min": 600.0,
            "threshold_max": 3071.0,
            "fill_holes": True,
            "close_radius": 1,
            "open_radius": 0,
            "min_component_mm3": 100.0,
            "label": "Dense bone",
            "color": "#f2f2f2",
        },
        "Higher threshold that isolates only the densest bone, useful for cortical shells.",
        "#f2f2f2",
    ),
    _preset(
        "ct_soft_tissue_body",
        "Body / soft tissue envelope",
        "threshold",
        ["CT", "CBCT"],
        {
            "mode": "absolute",
            "threshold_min": -300.0,
            "threshold_max": 2000.0,
            "fill_holes": True,
            "close_radius": 2,
            "open_radius": 1,
            "min_component_mm3": 5000.0,
            "keep_largest_n": 1,
            "label": "Body envelope",
            "color": "#7fd8c8",
        },
        "Captures everything denser than air, giving the patient outline as a single surface.",
        "#7fd8c8",
    ),
    _preset(
        "ct_lung",
        "Lung parenchyma",
        "threshold",
        ["CT", "CBCT"],
        {
            "mode": "absolute",
            "threshold_min": -1000.0,
            "threshold_max": -400.0,
            "fill_holes": False,
            "close_radius": 1,
            "open_radius": 0,
            "min_component_mm3": 500.0,
            "label": "Lung",
            "color": "#a8d8f0",
        },
        "Air-filled lung parenchyma. Note that the region touches the outside air, so 'fill holes' "
        "is disabled.",
        "#a8d8f0",
    ),
    _preset(
        "ct_region_grow_bone",
        "Bone via region growing (connected threshold)",
        "region_growing",
        ["CT", "CBCT"],
        {
            "seed_mode": "auto_largest",
            "seed_count": 1,
            "criterion": "absolute",
            "threshold_min": 200.0,
            "threshold_max": 3071.0,
            "fill_holes": True,
            "close_radius": 1,
            "open_radius": 0,
            "min_component_mm3": 50.0,
            "label": "Grown bone",
            "color": "#e8b04b",
        },
        "Uses the largest above-200 HU component as a seed and grows the connected bone structure "
        "only -- useful to separate the spine from adjacent ribs.",
        "#e8b04b",
    ),
    _preset(
        "mr_percentile_high",
        "High-intensity tissue (percentile)",
        "threshold",
        ["MR", "PT", "NM", "OT"],
        {
            "mode": "percentile",
            "lower_percentile": 97.0,
            "upper_percentile": 100.0,
            "fill_holes": True,
            "close_radius": 1,
            "open_radius": 0,
            "min_component_mm3": 100.0,
            "label": "High-intensity tissue",
            "color": "#c084fc",
        },
        "For MRI and other non-HU modalities, absolute thresholds are meaningless. This preset "
        "selects the brightest 3% of voxels in the volume.",
        "#c084fc",
    ),
    _preset(
        "mr_percentile_mid",
        "Intermediate-intensity tissue (percentile)",
        "threshold",
        ["MR", "PT", "NM", "OT"],
        {
            "mode": "percentile",
            "lower_percentile": 70.0,
            "upper_percentile": 99.5,
            "fill_holes": True,
            "close_radius": 1,
            "open_radius": 0,
            "min_component_mm3": 200.0,
            "label": "Intermediate-intensity tissue",
            "color": "#f0a8d8",
        },
        "Percentile band centred on the upper-middle intensity range of the volume.",
        "#f0a8d8",
    ),
    _preset(
        "custom",
        "Custom threshold",
        "threshold",
        ["CT", "CBCT", "MR", "PT", "NM", "OT"],
        {
            "mode": "absolute",
            "threshold_min": 0.0,
            "threshold_max": 1000.0,
            "fill_holes": True,
            "close_radius": 1,
            "open_radius": 0,
            "min_component_mm3": 50.0,
            "label": "Custom region",
            "color": "#8ab4f8",
        },
        "Start from an empty window and choose the range yourself.",
        "#8ab4f8",
    ),
)


def list_presets(modality: str | None) -> list[dict[str, Any]]:
    mod = (modality or "").upper()
    out: list[dict[str, Any]] = []
    for preset in PRESETS:
        entry = dict(preset)
        entry["applicable"] = mod in preset["modalities"] or "OT" in preset["modalities"]
        if not entry["applicable"]:
            entry["applicability_note"] = (
                f"This preset is defined for {', '.join(preset['modalities'])}; the current series is "
                f"{mod or 'unknown modality'}. It can still be applied manually."
            )
        out.append(entry)
    return out


def default_preset_key(modality: str | None) -> str:
    mod = (modality or "").upper()
    if mod in ("CT", "CBCT"):
        return "ct_bone"
    if mod == "MR":
        return "mr_percentile_high"
    return "custom"
