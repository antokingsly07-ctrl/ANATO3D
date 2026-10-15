# Segmentation

Code: `backend/app/services/segmentation/`
(`base.py`, `threshold.py`, `region_growing.py`, `ai.py`, `refine.py`,
`registry.py`, `pipeline.py`).

## 1. The engine abstraction

Every method implements one interface, `SegmentationEngine` (`base.py`):

```python
class SegmentationEngine:
    key: str                 # "threshold" | "region_growing" | "ai" | ...
    display_name: str
    description: str
    category: str            # "classical" | "ai"
    uses_hounsfield: bool
    parameters: list[ParameterSpec]

    def availability(self, volume) -> Availability: ...
    def describe(self) -> dict: ...
    def segment(self, volume, params, progress) -> SegmentationMask: ...
```

Adding a method (AI or otherwise) means implementing this interface and
registering it. The API, the UI, the pipeline, and the validation layer need no
changes.

Returned masks are validated (`validate_mask_non_empty`) before acceptance, and
are post-processed through the shared refinement step (§4).

## 2. Classical engines

### 2.1 `threshold`

Selects voxels inside a window. Two modes:

- **absolute** — `threshold_min`/`threshold_max` in modality units (HU for CT).
- **percentile** — `lower_percentile`/`upper_percentile` of the volume's own
  intensity distribution. This is essential for MRI/PET/NM where absolute
  numbers are meaningless.

### 2.2 `region_growing`

A connected-threshold grow implemented with a geodesic morphological
reconstruction (`skimage.morphology.reconstruction`), which grows the mask from
seed points through the "passable" (in-window) region:

- `seed_mode`: `auto_largest` (seed from the largest in-window component),
  `auto_centroid`, or `manual`.
- `criterion`: `absolute` or `percentile` (same meaning as threshold).
- `seed_count`, `threshold_min`, `threshold_max`.

This is how you separate a connected structure (e.g. the spine) from adjacent
touching structures (e.g. ribs) that a global threshold would merge.

## 3. The AI engine (and why it is honestly unavailable)

`ai.py` defines `AISegmentation`, a real MONAI adapter:

- It reports `availability()` by checking, in order: is `monai` importable, is
  `torch` importable, is `ANATO3D_AI_MODEL_PATH` set and valid? If anything is
  missing it returns `available: False` **with the exact reason**.
- When configured, it performs genuine sliding-window inference through
  `monai.bundle`, applies correct intensity scaling, and runs
  connected-component post-processing.

**No model weights are bundled.** Shipping a network without provenance and
validation would produce clinically meaningless output, so instead the engine is
explicit about its state. `GET /api/health` therefore reports
`ai_segmentation_available: false` and an `ai_unavailable_reason` until you
configure a bundle.

`ClassicalFallbackSegmentation` exists so that any "fall back to classical"
behaviour is *named* and *labelled*, and can never be mistaken for AI output.

### Enabling a MONAI bundle

```bash
pip install "monai[all]" torch
export ANATO3D_AI_MODEL_PATH=/path/to/bundle_root   # inference.json / model.pt
export ANATO3D_AI_INPUT_KEY=image                   # default: image
export ANATO3D_AI_OUTPUT_KEY=pred                   # default: pred
export ANATO3D_AI_DEVICE=cpu                        # or cuda
```

The bundle must expose a `sliding_window_inference`/`inferer`-compatible
`scripts.inference`, or a `model.pt` next to `inference.json`.

After setting these, `GET /api/segmentation/engines?volume_id=...` reports the
`ai` engine as available, and `POST /api/segmentation/run` with
`"engine": "ai"` will use it. Its output is labelled with the engine that
actually produced it (`engine` vs `requested_engine` in the response), so the
provenance is never ambiguous.

## 4. Refinement (shared post-processing)

All masks pass through `refine.py`:

| Step | Parameter |
| --- | --- |
| Fill interior holes | `fill_holes` |
| Morphological closing | `close_radius` |
| Morphological opening | `open_radius` |
| Remove small components | `min_component_mm3` (physical volume!) |
| Keep largest N components | `keep_largest_n` |

Small-component removal uses **physical volume**, not voxel count, so a threshold
behaves consistently across different spacings. After refinement the mask is
re-validated to be non-empty.

## 5. Anatomical presets

`registry.PRESETS` provides ready-made parameter sets (see the README table).
Each preset declares:

```python
{
  "key", "label", "engine", "modalities", "params",
  "description", "color"
}
```

`list_presets(modality)` annotates each with:

- `applicable: bool` — does this preset's modality list include the volume's
  modality?
- `applicability_note` — why not, when it doesn't.

Inapplicable presets remain selectable but are visibly caveated — the UI shows
the note rather than hiding the option, so a researcher is never blocked but is
always informed.

## 6. Parameter precedence

`POST /api/segmentation/run` merges, in increasing priority:

1. engine defaults,
2. the selected `preset`'s `params`,
3. the request's explicit `params` (only keys the caller actually sent).

So `{"preset": "ct_bone", "params": {"threshold_min": 250}}` keeps every bone
preset setting but overrides the threshold.

## 7. Response

```json
{
  "segmentation_id": "seg_…",
  "volume_id": "vol_…",
  "label": "Bone",
  "color": "#e8d5a8",
  "engine": "threshold",
  "requested_engine": null,
  "statistics": { "voxel_count": 20910, "volume_mm3": 18819.0,
                  "volume_cm3": 18.819, "components": 1 },
  "params": { "mode": "absolute", "threshold_min": 200.0, "…": "…" },
  "notes": ["…"], "warnings": ["…"], "findings": [],
  "duration_s": 0.31
}
```

`engine` is what actually ran; `requested_engine` is what was asked for — they
differ only when a fallback occurred, which is also recorded in `warnings`.

## 8. Limitations

- One label per segmentation; multi-label output is not modelled yet.
- Classical thresholding is sensitive to HU calibration and metal/beam-hardening
  artefacts.
- Percentile presets are heuristic for non-HU modalities.
- Region growing depends on seed selection; `auto_largest` can pick an
  unintended component if the largest in-window blob is not the target.
- No atlas/prior-based methods yet.
