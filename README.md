# ANATO3D

**Patient-specific 3D anatomical reconstruction from DICOM imaging data.**

ANATO3D is a research/teaching prototype that turns a stack of DICOM slices into a
navigable 3D anatomical model — with real spatial ordering, real volume
reconstruction, real segmentation, real marching-cubes meshing, real
measurements, and an honest validation layer that refuses to invent numbers it
cannot justify.

It ships as a FastAPI backend (all the imaging science) plus a React/TypeScript
workbench (multiplanar viewer, 3D surface viewer, live operation log, export
panel).

> **This is not a medical device.** It has not been cleared, certified or
> validated for clinical use. It must not be used for diagnosis, treatment
> planning, or any decision affecting a patient. See
> [What this is *not*](#2-what-this-is-not).

---

## Table of contents

1. [What this is](#1-what-this-is)
2. [What this is *not*](#2-what-this-is-not)
3. [Feature overview](#3-feature-overview)
4. [The 7-stage pipeline](#4-the-7-stage-pipeline)
5. [How spatial ordering works](#5-how-spatial-ordering-works)
6. [Reconstruction](#6-reconstruction)
7. [Segmentation engines and presets](#7-segmentation-engines-and-presets)
8. [Meshing and 3D export](#8-meshing-and-3d-export)
9. [Measurements and analysis](#9-measurements-and-analysis)
10. [Validation and accuracy metrics](#10-validation-and-accuracy-metrics)
11. [Architecture](#11-architecture)
12. [Repository layout](#12-repository-layout)
13. [Technology stack](#13-technology-stack)
14. [Getting imaging data](#14-getting-imaging-data)
15. [Running locally](#15-running-locally)
16. [Running with Docker](#16-running-with-docker)
17. [REST API reference](#17-rest-api-reference)
18. [Configuration](#18-configuration)
19. [Testing](#19-testing)
20. [Security, privacy and limits](#20-security-privacy-and-limits)
21. [Known limitations](#21-known-limitations)
22. [Troubleshooting](#22-troubleshooting)
23. [Roadmap / next improvements](#23-roadmap--next-improvements)

---

## 1. What this is

ANATO3D demonstrates a complete, honestly-implemented medical-imaging pipeline in
a form a student or researcher can read, run and extend:

- **Ingest** a DICOM series (or generate a labelled synthetic phantom).
- **Inspect** headers, group files into studies/series, and derive the true
  slice ordering from patient-space geometry.
- **Reconstruct** a 3D volume with a correct physical coordinate system
  (DICOM LPS, millimetre spacing, affine transform).
- **View** orthogonal planes (axial / coronal / sagittal) with real window/level
  applied to the raw pixel data.
- **Segment** anatomy with classical algorithms, an anatomical preset library,
  and a pluggable AI adapter (MONAI/nnU-Net) that is honest about being
  unconfigured rather than faking output.
- **Mesh** the segmentation with marching cubes, refine it, and view/export the
  surface as GLB / STL / OBJ.
- **Measure** the result in physical units (volumes, surface area, bounding box,
  centroid, principal axes, intensity statistics).
- **Validate** reconstruction and segmentation, and — *only when a reference
  mask actually exists* — compute Dice, IoU, Hausdorff and ASSD.
- **Export** volume, mask, meshes, reports and a single ZIP bundle.

Every number the UI shows is either read from real geometry or explicitly marked
as unavailable.

---

## 2. What this is *not*

These guarantees are enforced in code, not just in prose:

- **Not a diagnostic tool.** Nothing here is clinically validated. All output
  carries a disclaimer.
- **No fabricated metrics.** If no ground-truth reference mask is provided, the
  validation report states
  `"Ground truth unavailable — quantitative segmentation accuracy cannot be calculated."`
  and reports *no* Dice/IoU/Hausdorff values. It never invents them.
- **No filename-based slice ordering.** Slices are ordered by projecting each
  slice's `ImagePositionPatient` onto the series normal (with handedness
  correction). `InstanceNumber` is only ever a last-resort, opt-in fallback that
  raises a loud warning and flags the volume as *spatially unverified*.
- **No hidden failures.** Reconstruction problems surface as
  `ERROR` / `WARNING` / `INFO` findings in the UI and the validation report.
  When reconstruction may be unreliable, the app says so.
- **No fake demo data pretending to be real.** The only generated data is a
  clearly-labelled **synthetic** phantom, and it is never presented as a real
  patient.
- **No upload execution.** Uploaded files are stored as inert bytes and parsed
  with `pydicom`; they are never executed.
- **No AI theatre.** The AI engine reports *why* it is unavailable. It does not
  return a plausible-looking mask when it has no model.

---

## 3. Feature overview

| Area | What you get |
| --- | --- |
| DICOM ingest | Single files, multiple files, `.zip` archives; per-file error isolation; non-image and unreadable files reported, not ignored |
| Metadata | Study/series grouping, modality, rows/columns, spacing, orientation codes, patient position, reconstructability verdict |
| Ordering | Patient-space ordering, gaps, duplicates, spacing regularity, orientation consistency, oblique detection |
| Reconstruction | LPS affine, resampling to uniform/isotropic grids, HU rescaling, preprocessing, multi-plane extraction, PNG thumbnails |
| Segmentation | Threshold (absolute + percentile), region growing (geodesic), MONAI adapter, explicit classical fallback, morphology refinement |
| Meshing | Marching cubes, winding/label correction, padding to close border surfaces, decimation, smoothing, watertightness check |
| Export | NIfTI, NPZ, NPY, GLB, GLTF, STL, OBJ, JSON metadata, validation report, analysis report, ZIP bundle |
| Viewer | Raw int16/float32 slices with real W/L, orientation labels, mask overlay, R3F 3D surface viewer |
| UX | 7-step rail with real gating, backend-driven progress bar, streamed operation log, severity-coded findings |
| Jobs | Every heavy operation runs as a real job with stage, progress, message and logs; pollable |

---

## 4. The 7-stage pipeline

```
 Upload ──▶ Inspect ──▶ Reconstruct ──▶ Segment ──▶ 3D Model ──▶ Analyze ──▶ Export
   │          │             │             │            │            │          │
 DICOM or   headers,     volume +      mask +      surface +    measure-   files,
 synthetic  series,      geometry      stats       formats      ments +    reports,
 phantom    ordering                                          validation  bundle
```

Each stage maps to real backend endpoints and is gated: you cannot segment
before you have a volume, cannot mesh before you have a mask, and the UI will not
let you pretend otherwise.

---

## 5. How spatial ordering works

This is the part most naive implementations get wrong, so it is worth being
precise.

For a slice `i` we take the DICOM geometry:

- `ImageOrientationPatient` → two orthogonal direction cosines:
  - `row_dir` = first triplet (direction of increasing column index),
  - `column_dir` = second triplet (direction of increasing row index),
- `ImagePositionPatient` → `origin_i` (LPS, mm).

The **slice axis** is the cross product `slice_dir = row_dir × column_dir`
(a unit vector normal to the image plane).

Each slice is assigned a scalar position by projecting its origin onto the slice
axis:

```
projection_i = dot(origin_i, slice_dir)   # after normalising the origin
```

Slices are then **sorted by `projection_i`**, and the inter-slice spacing is the
median of the successive differences. Sorting by projection is orientation- and
handedness-correct and is completely independent of filenames and
`InstanceNumber`.

Additional checks run on the ordered stack:

- **Gaps** — the largest spacing is compared with the median; a significant
  deviation is reported as a possible missing slice.
- **Duplicates** — two slices projecting to the same location.
- **Irregular spacing** — e.g. a localiser mixed into the series.
- **Orientation consistency** — all slices must share the same
  `ImageOrientationPatient` within tolerance.
- **Obliquity** — the slice axis is compared with the anatomical axes; oblique
  acquisitions are labelled as such.

If a series has **no usable geometry at all**, reconstruction refuses:

```
ERROR  IncompleteOrientationError
       The series cannot be ordered from patient-space geometry.
       (hint: set allow_instance_number_fallback=true to proceed anyway)
```

Setting `allow_instance_number_fallback=true` proceeds using `InstanceNumber`,
but the volume is then reported as **spatially unverified** and every downstream
report carries that flag.

---

## 6. Reconstruction

The reconstruction stage builds a real 3D array plus the affine that maps voxel
indices to patient space.

- **Array convention**: `volume[k, j, i]` = (slice, row, column).
- **Affine**: `physical_lps = origin + direction @ (index * spacing)` where
  `direction = column_stack([slice_dir, row_dir, column_dir])`. This matches the
  ITK/GDCM convention (determinant `-1`, i.e. a right-handed voxel grid in a
  left-handed patient space).
- **Rescaling**: stored pixel values are converted to modality units via
  `HU = stored * RescaleSlope + RescaleIntercept` when the tags are present.
- **Resampling**: `resample_volume(array, spacing, target_spacing)` resamples to
  a uniform or isotropic grid. The new axis length is
  `max(1, round(span / target) + 1)` — derived from the physical extent, not a
  naive scale factor.
- **Preprocessing**: intensity normalisation, edge-preserving smoothing for the
  mesh path, and a voxel-count guard (`max_voxels_soft` / `max_voxels_hard`).

Plane extraction uses the volume's own axes:

| Plane | In-plane axes (columns, rows) | Notes |
| --- | --- | --- |
| Axial | `+X` (R→L), `+Y` (P→A) | top = A, bottom = P, left = R, right = L |
| Coronal | `+X` (R→L), `-Z` (S→I) | top = S, bottom = I, left = R, right = L |
| Sagittal | `+Y` (P→A), `-Z` (S→I) | top = S, bottom = I, left = A, right = P |

Orientation labels are computed from the volume's actual direction cosines and
shown on the viewer, so they stay correct for oblique or flipped acquisitions.

**Viewer space** (for the 3D viewport) is
`X_view = X_lps` (Left), `Y_view = Z_lps` (Superior), `Z_view = −Y_lps`
(Anterior) — the conventional radiological orientation.

---

## 7. Segmentation engines and presets

All engines implement one `SegmentationEngine` interface, so classical and AI
methods are interchangeable and future methods can be dropped in.

| Engine key | Name | Notes |
| --- | --- | --- |
| `threshold` | Threshold | Absolute HU window or percentile band; morphology cleanup |
| `region_growing` | Region growing | Geodesic reconstruction from auto or manual seeds (connected threshold) |
| `ai` | AI model (MONAI bundle) | Real MONAI sliding-window inference **when configured**; otherwise reports exactly why it is unavailable |
| `classical_fallback` | Classical fallback | Makes the fallback explicit and labels its output as classical |

### Presets

| Key | Label | Engine | Modalities |
| --- | --- | --- | --- |
| `ct_bone` | Bone (cortical + trabecular) | threshold | CT, CBCT |
| `ct_dense_bone` | Dense bone only | threshold | CT, CBCT |
| `ct_soft_tissue_body` | Body / soft tissue envelope | threshold | CT, CBCT |
| `ct_lung` | Lung parenchyma | threshold | CT, CBCT |
| `ct_region_grow_bone` | Bone via region growing | region_growing | CT, CBCT |
| `mr_percentile_high` | High-intensity tissue (percentile) | threshold | MR, PT, NM, OT |
| `mr_percentile_mid` | Intermediate-intensity tissue (percentile) | threshold | MR, PT, NM, OT |
| `custom` | Custom threshold | threshold | all |

Presets that do not match the volume's modality are still selectable, but the
API marks them `applicable: false` and attaches an `applicability_note` — the UI
shows the caveat rather than silently hiding the option.

Every mask is post-processed through a shared refinement step (hole filling,
morphological opening/closing, small-component removal, optional
"keep largest N") and validated to be non-empty before it is accepted.

### Enabling the AI engine

No model is bundled. To activate the MONAI path:

```bash
pip install "monai[all]" torch
export ANATO3D_AI_MODEL_PATH=/path/to/monai_bundle   # dir with inference.json / model.pt
export ANATO3D_AI_INPUT_KEY=image                     # default: image
export ANATO3D_AI_OUTPUT_KEY=pred                     # default: pred
export ANATO3D_AI_DEVICE=cpu                          # or cuda
```

Until then, `GET /api/health` returns `ai_segmentation_available: false` with a
reason string, and the engine reports the exact prerequisite that is missing.
See [`docs/segmentation.md`](docs/segmentation.md).

---

## 8. Meshing and 3D export

- **Extraction**: `skimage.measure.marching_cubes` on the mask's signed distance
  field, with explicit winding and label correction so normals face outward.
  The volume is padded first: padding **closes surfaces** at the volume border so
  the mesh is watertight, but it also means a structure that genuinely touches
  the acquisition boundary gets an artificial flat cap. The voxel volume remains
  the authoritative measurement; the mesh volume is a cross-check.
- **Refinement**: topological cleanup (merge duplicate vertices, drop degenerate
  and duplicated faces via trimesh), optional smoothing, and quadric decimation
  when `fast-simplification` is available (with a vertex-clustering fallback).
- **Formats**: GLB (viewer + interchange), GLTF, STL (3D printing), OBJ.
- **Checks**: watertightness, surface area, and volume comparison against the
  voxel volume.

Measurements carried forward include both the voxel-derived and mesh-derived
values so you can see them agree (or not).

---

## 9. Measurements and analysis

`GET /api/analysis/{mesh_id}` returns a structured analysis payload with:

- **Voxel metrics**: voxel count, voxel volume (mm³/cm³), bounding box size in
  physical units, centroid in patient space.
- **Mesh metrics**: triangle/vertex count, mesh surface area (mm²), mesh volume,
  watertightness.
- **Shape metrics**: principal-axis lengths from the covariance of the surface
  points, sphericity/compactness where meaningful.
- **Intensity statistics** within the structure: mean, std, min, max, percentiles.
- A **disclaimer** and any **warnings** that apply.

Physical-unit measurements are computed from the affine and spacing, so they are
in real millimetres regardless of the input grid.

---

## 10. Validation and accuracy metrics

Validation runs in two layers:

**1. Consistency and integrity checks** (always run, no ground truth needed):

- geometry is complete and non-degenerate,
- spacing is positive and finite, no zero/negative dimensions,
- orientation is consistent, no duplicate slice positions,
- volume dimensions match the series,
- segmentation is non-empty and lies inside the volume,
- mesh is watertight and its volume agrees with the voxel volume within
  tolerance,
- HU ranges are plausible for the declared modality.

Each check produces a `Finding` with `error` / `warning` / `info` severity, and a
summary with counts and an overall `status`.

**2. Quantitative accuracy metrics** (only when a reference mask exists):

When you upload a ground-truth mask (`.npz`, `.npy`, `.nii`/`.nii.gz`, `.zip`, or
a DICOM mask) — or use the synthetic phantom's bundled ground truth — the
backend computes:

| Metric | Meaning | Better when |
| --- | --- | --- |
| `dice` | Dice similarity coefficient | higher |
| `iou` | Intersection over union (Jaccard) | higher |
| `hausdorff` | Maximum symmetric surface distance (mm) | lower |
| `hausdorff95` | 95th-percentile Hausdorff distance (mm) | lower |
| `assd` | Average symmetric surface distance (mm) | lower |
| volume-difference metrics | Signed/absolute volume difference | near zero |

If no reference exists, the report is explicit:

```
Ground truth unavailable — quantitative segmentation accuracy cannot be calculated.
```

…and *no* metric values are emitted.

---

## 11. Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  Frontend (React + TypeScript + Vite + Tailwind + three.js)         │
│  • 7-step workflow state machine   • multiplanar canvas viewer      │
│  • window/level + mask overlay     • react-three-fiber 3D viewport  │
│  • job polling → progress + logs   • export panel, findings, log    │
└───────────────────────────────┬────────────────────────────────────┘
                                │  JSON / multipart over /api
┌───────────────────────────────▼────────────────────────────────────┐
│  Backend (FastAPI)                                                  │
│                                                                     │
│  api/routes   dicom · reconstruction · segmentation · mesh ·        │
│               validation · analysis · export · jobs · health        │
│  api/deps     store, job manager, settings, lookups                 │
│                                                                     │
│  services/                                                          │
│    dicom/          reader · series grouping · series_ordering       │
│    reconstruction/ volume (affine) · preprocessing · slices ·render │
│    segmentation/   base · threshold · region_growing · ai · refine  │
│                    registry (engines + presets) · pipeline          │
│    mesh/           extract (marching cubes) · refine · exporters    │
│    analysis/       measurements                                     │
│    validation/     checks · metrics · ground_truth · service        │
│    export/         volume/mask/report/mesh serialisers · bundle     │
│    synthetic/      labelled phantom generator (+ ground truth)      │
│                                                                     │
│  core/    errors · jobs (ProgressReporter) · storage · security ·   │
│           logging                                                   │
│  models/  domain (Volume, SegMask, …) · records (registries)        │
│  schemas/ pydantic request/response models                          │
└─────────────────────────────────────────────────────────────────────┘
```

Heavy operations are dispatched as **jobs**. A request either runs inline
(`?wait=true`) or returns a `JobEnvelope` immediately (`?wait=false`) that the
client polls via `GET /api/jobs/{job_id}`. Progress and log lines shown in the UI
come from the backend's own job records — there are no front-end timers
simulating work.

See [`docs/architecture.md`](docs/architecture.md) for the deep dive.

---

## 12. Repository layout

```
ANATO3D/
├─ backend/
│  ├─ app/
│  │  ├─ api/            routes, deps, presenters
│  │  ├─ core/           errors, jobs, storage, security, logging
│  │  ├─ models/         domain + records
│  │  ├─ schemas/        pydantic models
│  │  └─ services/       dicom, reconstruction, segmentation, mesh,
│  │                     analysis, validation, export, synthetic
│  ├─ scripts/           smoke_api.py, smoke_pipeline.py, smoke_segmentation.py
│  ├─ tests/             pytest suite (101 tests)
│  ├─ pytest.ini
│  ├─ requirements.txt
│  ├─ run.py             dev launcher
│  └─ Dockerfile
├─ frontend/
│  ├─ src/
│  │  ├─ api/            typed fetch client + job polling
│  │  ├─ components/     panels, viewers, console
│  │  ├─ lib/            formatting, window/level, mask decode
│  │  ├─ state/          workflow reducer + React provider
│  │  ├─ test/           vitest setup
│  │  └─ types.ts        API type mirror
│  ├─ package.json · vite.config.ts · tailwind.config.js
│  ├─ nginx.conf
│  └─ Dockerfile
├─ docs/                 architecture, dicom-processing, reconstruction,
│                        segmentation, validation, deployment
├─ sample_data/          how to obtain real + synthetic data
├─ scripts/              dev helper scripts
├─ docker-compose.yml
├─ .gitignore
└─ README.md
```

---

## 13. Technology stack

**Backend**

- Python ≥ 3.10, FastAPI, Uvicorn, Pydantic v2 / pydantic-settings
- `pydicom` — DICOM parsing
- `numpy`, `scipy` — array maths, distance transforms
- `scikit-image` — marching cubes, morphology, `morphology.reconstruction`
- `trimesh` (+ `fast-simplification`, `nibabel`, `numpy-stl`) — meshing and export
- `pytest`, `httpx` — testing

Deliberately **not** required: VTK/PyVista (heavy), SimpleITK, and
MONAI/torch (optional, gated behind configuration).

**Frontend**

- React 18, TypeScript, Vite 5, Tailwind CSS 3
- `three` + `@react-three/fiber` + `@react-three/drei` — 3D viewport
- `lucide-react` — icons
- Vitest + Testing Library + jsdom — tests

---

## 14. Getting imaging data

See [`sample_data/README.md`](sample_data/README.md) for the full list. In short:

**Public de-identified datasets (recommended):**

- **TCIA** — The Cancer Imaging Archive (<https://www.cancerimagingarchive.net/>)
- **LIDC-IDRI** — chest CT with lesion annotations
- **Medical Segmentation Decathlon** (<http://medicaldecathlon.com/>) — CT/MR with
  labels you can convert to a reference mask
- **NLST**, **Head-Neck-PET-CT**, **AAPM** low-dose CT — via TCIA
- **TotalSegmentator** reference segmentations

Always use de-identified data, and confirm the licence and any data-use
agreement before downloading.

**No data at hand?** Generate a synthetic phantom from the app (**Synthetic**
button in the Dataset panel) or the API:

```bash
curl -X POST http://127.0.0.1:8000/api/dicom/synthetic \
  -H "Content-Type: application/json" \
  -d '{"rows":256,"columns":256,"slices":96,"shuffle_filenames":true,"irregular_spacing":false}'
```

The phantom is written with a `.npz` ground-truth mask so the validation stage
has something real to compare against. It is **clearly labelled synthetic** and
is not a real anatomical image.

---

## 15. Running locally

### Backend

```bash
cd backend
python -m pip install -r requirements.txt
python run.py                 # http://127.0.0.1:8000
```

Interactive API docs: <http://127.0.0.1:8000/docs>

### Frontend (dev, with hot reload)

```bash
cd frontend
npm install
npm run dev                   # http://127.0.0.1:5173
```

The dev server proxies `/api` to `http://127.0.0.1:8000` (override with
`VITE_BACKEND_URL`).

### Single-process (production-style)

```bash
cd frontend && npm run build          # writes frontend/dist
cd ../backend && python run.py        # backend serves dist at /
```

Then open <http://127.0.0.1:8000>. The backend auto-mounts `frontend/dist` when
present.

### Suggested sample data

```bash
cd backend
python scripts/smoke_api.py           # end-to-end HTTP walk-through, prints metrics
```

---

## 16. Running with Docker

```bash
docker compose up --build
```

- backend → <http://127.0.0.1:8000>
- frontend → <http://127.0.0.1:5173> (nginx, proxies `/api` to the backend)

The backend container stores all working data under a named volume
(`anato3d-work`) so it is disposable. There is **no authentication** in the
compose setup — it is intended for local/offline use only. See
[`docs/deployment.md`](docs/deployment.md).

---

## 17. REST API reference

Base path: `/api`. All responses are JSON unless noted.

### System

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness, limits, AI availability |
| `GET` | `/api/jobs` | Recent jobs |
| `GET` | `/api/jobs/{job_id}` | One job's status, progress, logs, result |

### DICOM

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/dicom/upload` | Upload files/archive → `DatasetSummary` |
| `POST` | `/api/dicom/analyze` | Scan headers → studies, series, ordering (`?wait=`) |
| `POST` | `/api/dicom/synthetic` | Generate a labelled synthetic phantom |
| `GET` | `/api/dicom/{dataset_id}` | Stored dataset summary |

### Reconstruction

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/reconstruction/create` | Build the volume (`?wait=`, default false) |
| `POST` | `/api/reconstruction/preprocess` | Applied preprocessing steps |
| `GET` | `/api/reconstruction/window-presets` | Window/level presets for a modality |
| `GET` | `/api/reconstruction/{volume_id}` | Volume record + report |
| `GET` | `/api/reconstruction/{volume_id}/geometry` | Affine, spacing, dimensions, counts |
| `GET` | `/api/reconstruction/{volume_id}/slice` | Raw slice pixels (base64 int16/float32) |
| `GET` | `/api/reconstruction/{volume_id}/thumbnail` | 8-bit PNG thumbnail (dependency-free) |

### Segmentation

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/segmentation/engines` | Engines + availability |
| `GET` | `/api/segmentation/presets` | Presets for a modality |
| `POST` | `/api/segmentation/run` | Run segmentation (`?wait=`, default true) |
| `GET` | `/api/segmentation/{segmentation_id}` | Segmentation metadata + stats |
| `GET` | `/api/segmentation/{segmentation_id}/mask` | Bit-packed mask slice (base64) |
| `DELETE` | `/api/segmentation/{segmentation_id}` | Delete a segmentation |

### Mesh

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/mesh/generate` | Marching cubes + refine (`?wait=`, default false) |
| `GET` | `/api/mesh/{mesh_id}` | Mesh metadata, measurements, formats |
| `GET` | `/api/mesh/{mesh_id}/glb` | GLB binary (for the viewer) |
| `GET` | `/api/mesh/{mesh_id}/download?format=` | Download `glb`/`gltf`/`stl`/`obj` |

### Validation / analysis / export

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/validation/ground-truth` | Upload a reference mask (multipart) |
| `POST` | `/api/validation/run` | Checks + optional metrics (`?wait=`, default true) |
| `GET` | `/api/validation/{validation_id}` | Stored validation report |
| `GET` | `/api/analysis/{analysis_id}` | Measurements and statistics |
| `GET` | `/api/export/{dataset_id}/manifest` | List available exports |
| `GET` | `/api/export/{dataset_id}/file/{key}` | Download a single export or the bundle |

**Error shape** (all handled errors):

```json
{ "code": "dicom_error", "message": "Human-readable summary.", "detail": "Optional hint." }
```

HTTP status mapping: `400` validation, `404` not found, `415` unsupported file,
`413` too large, `422` DICOM/ordering error and schema validation, `501`
capability unavailable, `500` unexpected (details logged server-side only,
never returned to the client).

---

## 18. Configuration

All settings use the `ANATO3D_` prefix and can be set via environment variables
or a `.env` file in `backend/`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `ANATO3D_DATA_DIR` | `_work` | Root for all working data (wiped/disposable) |
| `ANATO3D_DEBUG` | `false` | Verbose logging |
| `ANATO3D_MAX_FILE_BYTES` | `536870912` | Max single upload |
| `ANATO3D_MAX_UPLOAD_BYTES` | `2147483648` | Max request total |
| `ANATO3D_MAX_UPLOAD_FILES` | `4000` | Max files per request |
| `ANATO3D_MAX_VOXELS_SOFT` | `220000000` | Above this, downsample before segmentation |
| `ANATO3D_MAX_VOXELS_HARD` | `900000000` | Refuse above this |
| `ANATO3D_WORKSPACE_TTL_MINUTES` | `240` | Idle workspace lifetime |
| `ANATO3D_CORS_ORIGINS` | dev origins | Allowed browser origins |
| `ANATO3D_HOST` / `ANATO3D_PORT` | `127.0.0.1` / `8000` | Dev launcher bind |
| `ANATO3D_AI_MODEL_PATH` | *(unset)* | MONAI bundle → enables the AI engine |

Frontend: `VITE_API_BASE` (default same-origin), `VITE_BACKEND_URL` (dev proxy
target).

---

## 19. Testing

**Backend** (real behaviour, not smoke-only):

```bash
cd backend
python -m pytest -q
# 101 passed
```

The suite covers DICOM ingest, ordering (including shuffled filenames,
gaps, duplicates, missing geometry and the opt-in fallback), volume affine
correctness, plane extraction and labels, segmentation engines and refinement,
marching-cubes meshing and decimation, ground-truth metrics, export formats,
security (path traversal, size limits, filename sanitisation, no internals
leaked) and the full HTTP flow including async job polling and error cases.

**Frontend**:

```bash
cd frontend
npm test          # 41 passed
npm run typecheck
npm run build
```

Covers number/size/duration formatting, slice decoding and window/level maths
(including bit-packed mask decode and overlay), the workflow reducer and step
gating, API error normalisation and job polling, and the slice viewer's fetch,
render and error paths.

---

## 20. Security, privacy and limits

- **Uploads are inert.** Files are stored as bytes and parsed with `pydicom`.
  Nothing is executed. Archives are extracted with path-traversal protection.
- **Filenames are sanitised** and all writes go through a `safe_join` helper;
  the server never writes outside a job's workspace.
- **Isolation.** Each upload gets a unique dataset id and its own workspace
  directory; each job gets a unique id.
- **Limits.** Per-file, per-request and file-count limits are enforced before
  and during reading (streamed, so a lying `Content-Length` cannot exhaust
  memory).
- **No patient data in logs.** Raw pixel data and identifying tags are never
  logged; logs contain counts, dims and timings.
- **No stack traces to users.** Unexpected errors return a generic message; the
  traceback is written to the server log only.
- **Privacy.** Everything runs locally/offline. The prototype performs no
  telemetry and makes no outbound network calls.
- **Reaper.** Idle workspaces are deleted automatically
  (`workspace_ttl_minutes`).

---

## 21. Known limitations

- **Not clinically validated.** Metrics are only as meaningful as the reference
  mask you supply.
- **Border capping.** Padding closes surfaces at the volume boundary, so a
  mask touching the border gets an artificial flat cap. Voxel volume is the
  authoritative measurement.
- **No GPU dependency.** Segmentations are classical and fast; the AI path needs
  a user-supplied model bundle.
- **Single structure at a time.** The pipeline processes one label per
  segmentation; multi-label output is not yet modelled.
- **No full DICOM series browser.** The viewer shows orthogonal planes, not a
  scrollable cine of every acquisition.
- **No registration/fusion.** Multiple series are not co-registered.
- **Prototype-scale guards.** Very large volumes are downsampled before
  segmentation to stay interactive; the report says when this happened.
- **Synthetic phantom is not anatomy.** It exists to exercise the pipeline and
  to provide a ground truth for the validation demo.

---

## 22. Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `backend offline` badge | Backend not running, or wrong `VITE_BACKEND_URL`. Start `python run.py`. |
| 422 on reconstruct | Series lacks usable geometry. Inspect findings, or opt in to `allow_instance_number_fallback` (it will be flagged). |
| `AI unavailable` | Expected unless `monai`+`torch` are installed **and** `ANATO3D_AI_MODEL_PATH` points at a bundle. |
| Empty 3D viewport | Generate a mesh first; the "3D surface" tab is disabled until one exists. |
| Ground-truth state `unavailable` | No reference mask was uploaded; upload one to enable accuracy metrics. |
| `413` on upload | Raise `ANATO3D_MAX_FILE_BYTES` / `ANATO3D_MAX_UPLOAD_BYTES`, or split the transfer. |
| Slow segmentation | Volume exceeded `max_voxels_soft`; reduce the series or enable isotropic/coarser resampling. |

---

## 23. Roadmap / next improvements

- Multi-label segmentation and per-label meshes.
- A real segmentation model (nnU-Net / MONAI bundle) shipped as an optional
  download with documented provenance.
- Registration of multiple series and fusion views.
- A scrollable cine viewer and MPR crosshair linking.
- More validation metrics (surface Dice, per-region tables) and ROC where
  applicable.
- Async task queue (Celery/RQ) and object storage for multi-user deployments.
- Authentication, audit logging and role separation for any shared deployment.
- DICOMweb (QIDO/WADO) ingest in addition to file upload.

---

### Disclaimer

ANATO3D is a research and educational prototype. It is **not** a certified
medical device and must not be used for clinical decision-making. Always consult
qualified professionals and use appropriate, de-identified data in accordance
with all applicable regulations and data-use agreements.
