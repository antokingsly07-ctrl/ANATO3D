# Validation

Code: `backend/app/services/validation/`
(`checks.py`, `metrics.py`, `ground_truth.py`, `service.py`).

ANATO3D validation has a strict rule: **it never presents a number it cannot
justify.** Validation therefore has two independent layers.

```
run_validation(dataset, analysis, series, ordering, volume, segmentation,
               reference_mask, reference_source, sections)
```

## 1. Layer 1 — consistency and integrity checks (always run)

These require no ground truth and always produce findings.

| Check | What it verifies |
| --- | --- |
| Geometry | orientation present, spacing positive and finite, ≥2 slices, no zero/negative dimensions |
| Ordering | no duplicate slice positions, no gaps (or gaps reported), consistent orientation, obliquity labelled |
| Volume | dimensions match the series, affine is finite and non-degenerate, intensity range plausible |
| Segmentation | mask is non-empty, lies within the volume bounds, no absurd volume for the modality |
| Mesh | surface is watertight, mesh volume agrees with the voxel volume within tolerance |
| Provenance | instance-number fallback, resampling and dtype downcast are surfaced |

Each check yields a `Finding`:

```json
{
  "code": "ordering_gap",
  "severity": "warning",
  "title": "Possible missing slice",
  "message": "…",
  "context": { "…": "…" }
}
```

`severity` ∈ `error | warning | info`. `checks.summarise()` aggregates them into:

```json
{ "total": 7, "errors": 0, "warnings": 2, "info": 5, "status": "warning" }
```

`status` is `error` if any error exists, else `warning` if any warning exists,
else `ok`.

## 2. Layer 2 — quantitative accuracy metrics (only with ground truth)

A reference mask can come from:

- **uploading one** via `POST /api/validation/ground-truth` (multipart:
  `dataset_id` + `file`), or
- the **synthetic phantom's** bundled `.npz` ground truth.

### Accepted reference formats

| Format | Notes |
| --- | --- |
| `.npz` | ANATO3D mask archive: `mask` (bit-packed `uint8`) + `shape`, or `array` |
| `.npy` | boolean/integer array |
| `.nii` / `.nii.gz` | NIfTI (requires `nibabel`) |
| `.zip` | archive of a DICOM mask series |
| `.dcm` / no extension | a DICOM mask series |

Anything else is rejected with a clear message. If the reference shape differs
from the segmentation shape, it is resampled by nearest neighbour **and that is
always reported** as a note/warning — never silently.

### Metrics

| Key | Label | Unit | Better |
| --- | --- | --- | --- |
| `dice` | Dice similarity coefficient | 0–1 | higher |
| `iou` | Intersection over Union (Jaccard) | 0–1 | higher |
| `precision` | Precision | 0–1 | higher |
| `recall` | Recall (sensitivity) | 0–1 | higher |
| `hausdorff` | Hausdorff distance (HD) | mm | lower |
| `hausdorff95` | 95th percentile Hausdorff (HD95) | mm | lower |
| `assd` | Average symmetric surface distance | mm | lower |
| `volume_prediction` | Predicted volume | cm³ | neutral |
| `volume_reference` | Reference volume | cm³ | neutral |
| `volume_difference` | Volume difference (pred − ref) | cm³ | neutral |
| `relative_volume_difference` | Relative volume difference | % | neutral |

### How distances are computed

Boundary metrics use the **surface voxels** of each mask (a binary erosion
subtraction) and an exact Euclidean distance transform
(`scipy.ndimage.distance_transform_edt`) sampled with the **true anisotropic
voxel spacing**, so every distance is in millimetres of real anatomy:

- `HD` = the maximum of the two directed maximum distances,
- `HD95` = the 95th percentile of each directed distance distribution, then the
  maximum of the two — the standard robust choice,
- `ASSD` = mean of the combined symmetric surface distances.

The report includes a per-count confusion breakdown
(`true_positive`, `false_positive`, `false_negative`, `true_negative`) so
over/under-segmentation is visible directly.

### Edge cases handled honestly

- **Both masks empty** → a warning ("overlap metrics are undefined"), no values.
- **One mask empty** → overlap metrics still computed; boundary distances
  explicitly skipped with a warning.
- **Shape mismatch** → resample + report; if impossible, raise with a clear
  message.

## 3. When there is no ground truth

`unavailable_report()` returns exactly:

```json
{
  "state": "unavailable",
  "message": "Ground truth unavailable - quantitative segmentation accuracy cannot be calculated.",
  "detail": "No reference (ground-truth) mask was supplied …",
  "metrics": [],
  "counts": null,
  "notes": [],
  "warnings": []
}
```

The UI shows this verbatim (with a warning tone) and displays **no metric
values**. There is no fallback that quietly substitutes a plausible number.

## 4. Validation response

```json
{
  "validation_id": "val_…",
  "summary": { "total": 9, "errors": 0, "warnings": 2, "info": 7, "status": "warning" },
  "findings": [ … ],
  "ground_truth": {
    "state": "computed",
    "message": "Ground-truth metrics computed against 'phantom_ground_truth.npz'.",
    "reference_source": "phantom_ground_truth.npz",
    "metrics": [ { "key": "dice", "label": "…", "value": 0.874164, "unit": "0-1",
                   "interpretation": "…", "direction": "higher_is_better" }, … ],
    "counts": { "true_positive": …, "false_positive": …, "false_negative": …, "true_negative": … },
    "notes": [ … ],
    "warnings": [ … ]
  },
  "dataset_information": { … },
  "interpretation_notes": [ … ],
  "disclaimer": "…",
  "duration_s": 0.14
}
```

## 5. Interpreting the numbers

- **Dice / IoU** measure overlap; Dice is more forgiving of small structures.
- **Precision vs recall** tell you the *direction* of the error:
  low precision → over-segmentation, low recall → under-segmentation.
- **HD95 / ASSD** measure boundary agreement in millimetres; HD is dominated by
  single outliers, which is why HD95 is reported alongside it.
- **Volume difference** is signed, so you can see whether the segmentation runs
  large or small.

A high Dice with a large HD95 means the volumes overlap well but the surface is
sloppy — a distinction that matters for 3D-printing or surface-based analysis.

## 6. What validation is *not*

- It is **not** a clinical accuracy claim. Metrics are only as meaningful as the
  reference mask you provide.
- It does **not** attempt to hide failures. Every consistency problem becomes a
  visible finding.
- It does **not** invent reference data. The only built-in reference is the one
  the synthetic phantom honestly ships with.

## 7. Reference: the synthetic phantom

The phantom writes a ground-truth `.npz` alongside its slices, which is why
`python scripts/smoke_api.py` reports real numbers (Dice ≈ 0.874, IoU ≈ 0.776,
HD95 = 8.0 mm). That ground truth reflects the *synthetic* lesion/bone geometry,
not a patient — it exists to prove the metric pipeline works end to end.
