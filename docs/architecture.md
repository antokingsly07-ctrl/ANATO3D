# Architecture

ANATO3D is deliberately split into a **science backend** (FastAPI) and a
**presentation frontend** (React). The backend owns every anatomical decision;
the frontend owns only how it is displayed and which request to send next. This
keeps the imaging logic testable in isolation and means the UI can never
"invent" a result.

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Frontend                                                                    │
│  api/client.ts        typed fetch + ApiRequestError + job polling           │
│  state/workflow.ts    pure reducer: steps, artefacts, logs, notices         │
│  state/WorkflowProvider.tsx  binds reducer ↔ API, owns job lifecycle        │
│  components/*         panels, SliceViewer (canvas), Viewport3D (three.js)   │
└───────────────────────────────┬────────────────────────────────────────────┘
                                │ HTTP JSON / multipart / binary
┌───────────────────────────────▼────────────────────────────────────────────┐
│ Backend                                                                     │
│  app/main.py          app factory, exception handlers, static mount         │
│  app/api/routes/*     thin HTTP layer (validation + shaping only)           │
│  app/api/deps.py      registries, lookups, settings, job manager            │
│  app/api/presenters.py  domain → response shaping                           │
│  app/services/*       the actual science                                    │
│  app/core/*           errors, jobs, storage, security, logging              │
│  app/models/*         domain objects + in-memory records                    │
│  app/schemas/*        pydantic request/response contracts                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Layering rules

1. **Routes never do science.** A route parses the request, resolves ids via
   `deps`, calls a service, and shapes the response with a presenter. All maths
   lives in `services/`.
2. **Services are framework-free.** They import numpy/scipy/skimage/pydicom/
   trimesh, not FastAPI. That is why they can be tested directly.
3. **Domain objects are explicit.** `Volume`, `SegmentationMask`,
   `MeshAsset`, `OrderingResult`, `ReconstructionResult` carry their geometry and
   provenance; services do not pass bare arrays around loosely.
4. **Errors are typed.** `core.errors.AnatoError` subclasses carry an
   `http_status` and `to_dict()`. A single handler maps them to the public
   `{code, message, detail}` payload.

## Request/job model

Heavy work runs as a **job** so the HTTP loop is never blocked:

- `POST` endpoints accept `?wait=true|false`.
  - `wait=true` runs inline (`JobManager.run_sync`) and returns the completed
    envelope — used for fast operations (analyze, segmentation) and by tests.
  - `wait=false` submits to a `ThreadPoolExecutor` (`JobManager.submit`) and
    returns a `JobEnvelope` with `status: "pending"|"running"`.
- The client either polls `GET /api/jobs/{job_id}` or the frontend's `pollJob`
  helper does it automatically, streaming `progress` and `logs` into the UI.
- A synchronous job that fails returns HTTP **200** with
  `status: "failed"` and `error: {...}` — the request itself was valid. Genuine
  request problems (bad id, bad parameter) raise *before* dispatch and produce a
  real HTTP error code.

`ProgressReporter` is threaded through the services and calls `.step(stage,
fraction, message)` at **real checkpoints** (e.g. after each batch of slices).
Progress is never synthesised from elapsed time.

### Status codes

| Error class | HTTP |
| --- | --- |
| `ValidationError` | 400 |
| `NotFoundError` | 404 |
| `PayloadTooLargeError` | 413 |
| `UnsupportedFileError` | 415 |
| `DicomError` / `NoValidDicomSeriesError` | 422 |
| Pydantic `RequestValidationError` | 422 (`invalid_request`) |
| `SegmentationUnavailableError` | 501 |
| unhandled | 500 (generic message; traceback logged only) |

## Storage

All data lives under `ANATO3D_DATA_DIR` (default `backend/_work`):

```
_work/
└─ datasets/<dataset_id>/
   ├─ original/          inert uploaded bytes (opaque names)
   ├─ derived/           reconstructions, masks, meshes, reports, ground truth
   └─ dataset.json       the DatasetSummary record
```

- Every dataset id and job id is generated (`uuid4`), so collisions and
  traversal are structurally impossible.
- Writes go through `core.security.safe_join`, which refuses any path escaping
  the workspace.
- A background **reaper** thread deletes workspaces idle longer than
  `workspace_ttl_minutes` and prunes stale jobs.

## Domain conventions

These are the invariants the whole codebase depends on:

- **Volume array** `volume[k, j, i]` = (slice, row, column).
- **Affine** `physical_lps = origin + direction @ (index * spacing)` with
  `direction = column_stack([slice_dir, row_dir, column_dir])`, determinant `-1`
  (ITK/GDCM handedness).
- **Viewer space** `X_view = X_lps`, `Y_view = Z_lps`, `Z_view = -Y_lps`.
- **Plane screen axes**: axial `+X/+Y`, coronal `+X/-Z`, sagittal `+Y/-Z`.
- **Slice delivery**: raw `int16`/`float32` base64 for the interactive plane;
  dependency-free 8-bit PNG for thumbnails; bit-packed (`np.packbits`) base64 for
  masks.

## Frontend state

`state/workflow.ts` is a **pure reducer** — easy to unit-test without React:

- `WorkflowState` holds the active step and every artefact (dataset, analyze,
  volume, geometry, segmentation, mesh, validation, analysis, manifest), plus
  view state (plane, slice index, window/level) and the log/notice buffers.
- `stepStatus(state, key)` derives `locked | available | complete` from what
  actually exists. That single function drives the step rail, so gating cannot
  drift from reality.
- `WorkflowProvider` binds the reducer to the API: it starts jobs, polls them,
  and dispatches real backend log lines and progress into the store.

Because the reducer and the API client are separated, both are tested directly
(the reducer with pure calls, the client with a mocked `fetch`).

## Testing strategy

- **Backend (101 tests)**: exercise services directly (ordering, affine, planes,
  segmentation, meshing, metrics, export) plus the HTTP surface through
  `TestClient` (async polling, error codes, no internal leakage).
- **Frontend (41 tests)**: formatting, window/level + mask decode, reducer/step
  gating, API error normalisation and job polling, and the slice viewer's
  fetch/render/error paths.

The intent is that every test asserts *behaviour* (a correct affine, a real
Dice score, a rejected traversal) rather than mere existence.
