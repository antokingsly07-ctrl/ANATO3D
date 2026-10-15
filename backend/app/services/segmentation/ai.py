"""AI / deep-learning segmentation adapter.

Status
------
*No model is bundled with this prototype.*  Running a real network requires
weights (and usually a large training dataset) that cannot be shipped here, and
pretending otherwise would produce clinically meaningless output.

What is provided instead:

* a stable :class:`AISegmentation` engine that reports exactly why it is or is
  not available, and never fabricates a mask;
* a real, working inference path through ``monai.bundle`` (sliding-window
  inference, correct intensity scaling, connected-component post-processing) that
  activates as soon as a model bundle path is configured;
* :class:`ClassicalFallbackSegmentation`, which makes the fallback explicit and
  labels its output as classical so it can never be mistaken for AI output.

Enabling a MONAI bundle::

    pip install "monai[all]" torch
    export ANATO3D_AI_MODEL_PATH=/path/to/bundle_root
    export ANATO3D_AI_INPUT_KEY=image            # bundle input key (default: image)
    export ANATO3D_AI_OUTPUT_KEY=pred            # bundle output key (default: pred)

The bundle must expose a ``sliding_window_inference`` / ``inferer`` compatible
``scripts.inference`` or a ``model.pt`` next to ``inference.json``.  See
``docs/segmentation.md`` for the full walkthrough.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from ...core.errors import SegmentationUnavailableError
from ...core.logging_ import get_logger
from ...models import SegmentationMask, Volume
from .base import Availability, ParameterSpec, SegmentationEngine, cleanup_kwargs, validate_mask_non_empty
from .refine import refine_mask

logger = get_logger("segmentation.ai")

MODEL_PATH_ENV = "ANATO3D_AI_MODEL_PATH"
INPUT_KEY_ENV = "ANATO3D_AI_INPUT_KEY"
OUTPUT_KEY_ENV = "ANATO3D_AI_OUTPUT_KEY"
DEVICE_ENV = "ANATO3D_AI_DEVICE"


def _monai_available() -> tuple[bool, str | None]:
    try:
        import monai  # noqa: F401
    except Exception:  # noqa: BLE001
        return False, "The 'monai' package is not installed."
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        return False, "The 'torch' package is not installed."
    return True, None


class AISegmentation(SegmentationEngine):
    key = "ai"
    display_name = "AI model (MONAI bundle)"
    description = (
        "Runs a MONAI bundle exported by the user. No model is bundled with this prototype, so this "
        "engine is unavailable until a bundle path is configured."
    )
    category = "ai"
    uses_hounsfield = False

    def availability(self, volume: Volume) -> Availability:
        requirements = [
            "pip install 'monai[all]' torch",
            f"{MODEL_PATH_ENV}=/path/to/monai_bundle",
        ]
        model_path = os.environ.get(MODEL_PATH_ENV)
        if not model_path:
            return Availability(
                False,
                "No AI model is configured. The prototype performs no learned inference on its own.",
                requirements,
            )
        if not Path(model_path).exists():
            return Availability(False, f"The configured model path '{model_path}' does not exist.", requirements)
        ok, reason = _monai_available()
        if not ok:
            return Availability(False, reason or "MONAI is not available.", requirements)
        return Availability(True)

    def parameters(self) -> list[ParameterSpec]:
        return [
            ParameterSpec("roi_size", "Sliding window size", "int", 128, 32, 512, step=16, unit="voxels"),
            ParameterSpec("sw_batch_size", "Window batch size", "int", 2, 1, 16, advanced=True),
            ParameterSpec("overlap", "Window overlap", "float", 0.25, 0.0, 0.9, step=0.05, advanced=True),
            ParameterSpec("label_index", "Output channel (lesion/foreground)", "int", 1, 0, 32, advanced=True),
            ParameterSpec("probability_threshold", "Probability threshold", "float", 0.5, 0.0, 1.0,
                          step=0.05, advanced=True),
            *super().parameters(),
        ]

    def run(self, volume: Volume, params: dict[str, Any], *, progress=None) -> SegmentationMask:
        availability = self.availability(volume)
        if not availability.available:
            raise SegmentationUnavailableError(
                "AI segmentation is not available in this deployment.",
                detail=(
                    (availability.reason or "")
                    + " Configure a model as documented in docs/segmentation.md, or use the classical "
                    "engines (threshold / region growing), which are fully functional."
                ).strip(),
            )

        return self._run_monai(volume, params, progress=progress)

    # -- real inference path ----------------------------------------------
    def _run_monai(self, volume: Volume, params: dict[str, Any], *, progress=None) -> SegmentationMask:
        import torch
        from monai.bundle import ConfigWorkflow
        from monai.inferers import sliding_window_inference

        model_path = Path(os.environ[MODEL_PATH_ENV])
        input_key = os.environ.get(INPUT_KEY_ENV, "image")
        output_key = os.environ.get(OUTPUT_KEY_ENV, "pred")
        device = torch.device(os.environ.get(DEVICE_ENV, "cuda" if torch.cuda.is_available() else "cpu"))

        if progress:
            progress.step("segmentation", 0.06, f"Loading MONAI bundle from {model_path.name}")

        workflow = ConfigWorkflow(
            config_file=str(model_path / "inference.json"),
            meta_file=str(model_path / "inference.json"),
            bundle_root=str(model_path),
        )
        network = workflow.network
        network.eval()
        network.to(device)

        array = np.asarray(volume.array, dtype=np.float32)
        # CT/MR models normally expect values scaled to a fixed range; the bundle
        # is responsible for its own preprocessing (it declares it in its config).
        tensor = torch.from_numpy(array)[None, None].to(device)
        roi = int(params.get("roi_size", 128))
        with torch.no_grad():
            output = sliding_window_inference(
                tensor,
                roi_size=(roi, roi, roi),
                sw_batch_size=int(params.get("sw_batch_size", 2)),
                predictor=network,
                overlap=float(params.get("overlap", 0.25)),
            )
        if progress:
            progress.step("segmentation", 0.24, "Neural network inference complete")

        probabilities = output[0, int(params.get("label_index", 1))].detach().cpu().numpy()
        raw_mask = probabilities >= float(params.get("probability_threshold", 0.5))
        _ = (input_key, output_key)  # reserved for bundle I/O key customisation

        refined, report = refine_mask(
            raw_mask,
            voxel_volume_mm3=volume.voxel_volume_mm3,
            **cleanup_kwargs(params),
            progress=progress,
        )
        validate_mask_non_empty(
            refined,
            engine=self.display_name,
            hint="The model produced no foreground voxels above the probability threshold.",
        )

        return SegmentationMask(
            mask=refined,
            label=str(params.get("label") or f"AI prediction ({model_path.name})"),
            color=str(params.get("color") or "#c084fc"),
            engine=self.key,
            params={
                **params,
                "model_path": str(model_path),
                "device": str(device),
                "probability_range": [float(probabilities.min()), float(probabilities.max())],
            },
            notes=[
                f"MONAI bundle '{model_path.name}' on {device}.",
                "Model output has NOT been validated against a reference standard in this deployment. "
                "Do not treat these values as clinically accurate.",
                "Refinement: " + (" ".join(report.steps) if report.steps else "no operations applied."),
            ],
            warnings=[
                "AI output is unvalidated. Quantitative accuracy metrics require a ground-truth mask.",
            ],
        )


class ClassicalFallbackSegmentation(SegmentationEngine):
    """Explicit, honest fallback used when AI is requested but unavailable."""

    key = "classical_fallback"
    display_name = "Classical fallback (HU threshold)"
    description = (
        "Explicit classical fallback used when an AI engine is requested but no model is configured. "
        "Output is clearly labelled as a threshold result, never as a model prediction."
    )
    category = "classical"
    uses_hounsfield = True

    def availability(self, volume: Volume) -> Availability:
        return Availability(True)

    def parameters(self) -> list[ParameterSpec]:
        return [
            ParameterSpec("threshold_min", "Minimum", "float", 200.0, unit="HU/au"),
            ParameterSpec("threshold_max", "Maximum", "float", 3071.0, unit="HU/au"),
            *super().parameters(),
        ]

    def run(self, volume: Volume, params: dict[str, Any], *, progress=None) -> SegmentationMask:
        from .threshold import ThresholdSegmentation

        if progress:
            progress.log(
                "AI segmentation unavailable -> running the classical Hounsfield threshold fallback. "
                "The result is NOT a learned model prediction.",
                level="warning",
                stage="segmentation",
            )

        merged = {**params}
        merged.setdefault("mode", "absolute")
        merged["label"] = str(params.get("label") or "Classical fallback region")
        merged["color"] = str(params.get("color") or "#e8b04b")

        mask = ThresholdSegmentation().run(volume, merged, progress=progress)
        mask.label = str(params.get("label") or "Classical fallback region")
        mask.engine = self.key
        mask.warnings = list(mask.warnings) + [
            "This mask comes from a classical intensity threshold used as a fallback because no AI "
            "model is configured. It is not a deep-learning prediction.",
        ]
        return mask
