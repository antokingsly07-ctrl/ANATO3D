# Reconstruction

This document explains how an ordered DICOM stack becomes a 3D volume with a
physically correct coordinate system, and how planes are extracted from it.

Code: `backend/app/services/reconstruction/`
(`volume.py`, `preprocessing.py`, `slices.py`, `render.py`).

## 1. Orientation and spacing

From the ordered series (`OrderingResult`, see `dicom-processing.md`) we take:

- `row_dir` (columns increase) and `column_dir` (rows increase) from
  `ImageOrientationPatient`,
- `slice_dir = normalize(row_dir × column_dir)`,
- `origin` — the position of the first slice,
- `spacing = (slice_spacing, row_spacing, column_spacing)` in mm.

The **direction matrix** is built as:

```
direction = column_stack([slice_dir, row_dir, column_dir])
```

i.e. its columns are the physical steps taken when the *slice*, *row*, and
*column* indices each increase by one. Its determinant is `-1`, which is exactly
what ITK/GDCM produce for a right-handed voxel lattice expressed in the
left-handed DICOM/LPS patient space. We keep that convention rather than
"fixing" the sign, so exported volumes line up with standard tools.

## 2. The affine

For a voxel index `(k, j, i)` (slice, row, column):

```
physical_lps = origin + direction @ (index * spacing)
```

This is exposed directly by `GET /api/reconstruction/{volume_id}/geometry`
(`origin_lps_mm`, `direction`, `spacing_mm`, `dimensions`) and is what every
measurement, plane label, and export is derived from.

## 3. Building the volume

`build_volume(...)` produces a `ReconstructionResult`:

- `volume` — the array, shape `(K, J, I)`;
- `ordering` — the `OrderingResult` used;
- `frames` — per-slice frames for reference;
- `report` — a human-readable reconstruction report (§6).

The array dtype is chosen by `_pick_dtype`: `float32` for small volumes
(< 60M voxels) to preserve HU fidelity, otherwise a compact integer type to keep
memory bounded.

## 4. Intensity rescaling

Stored pixel values are converted to modality units when the tags are present:

```
value = stored * RescaleSlope + RescaleIntercept
```

For CT this yields Hounsfield Units; the volume's `intensity_units` records
`"HU"` (or the modality-appropriate unit) so the UI can label the window/level
readout correctly.

## 5. Resampling

`resample_volume(array, spacing, target_spacing) -> (array, spacing)`

Returns a **tuple**: the resampled array *and* its new spacing. Requests can ask
for a uniform grid (`uniform_spacing_mm = [s, r, c]`) or an isotropic one
(`isotropic: true`, target 1 mm).

The new axis length is derived from the **physical extent**, not a naive scale
factor:

```
n_new = max(1, round(span_mm / target_mm) + 1)
```

This avoids off-by-one shrink/grow errors that a simple `n * (s/target)` would
introduce. Resampled volumes are flagged (`is_resampled: true`) and the
reconstruction report notes the change.

## 6. Reconstruction report

`build_reconstruction_report(...)` records, among others:

| Key | Meaning |
| --- | --- |
| `slice_count` | source slice count |
| `reconstructed_slice_count` | slices in the volume |
| `dimensions` | `[K, J, I]` |
| `dimensions_order` | `"slices x rows x columns"` |
| `voxel_spacing_mm` | `[slice, row, column]` |
| `voxel_spacing_order` | `"slice x row x column"` |
| `orientation` | AXIAL/CORONAL/SAGITTAL/OBLIQUE |
| `ordering_method` | geometry vs fallback |
| `is_normalized`, `is_resampled` | preprocessing flags |
| `warnings` | anything that affects reliability |

The report is attached to the volume record so the UI and the export bundle can
show exactly how the volume was produced.

## 7. Plane extraction

`slices.py` implements the multiplanar views.

The `plane_axes()` helper maps a plane to the volume axes it uses:

| Plane | `slice_axis` | image rows come from | image columns come from |
| --- | --- | --- | --- |
| axial | `0` (K) | `+Y` (P→A) | `+X` (R→L) |
| coronal | `1` (J) | `-Z` (S→I) | `+X` (R→L) |
| sagittal | `2` (I) | `-Z` (S→I) | `+Y` (P→A) |

`extract_plane(volume, plane, index, with_array=True)` returns a `PlaneView`
carrying the array, its mm spacing, the position along the slice axis, and the
orientation **labels** (`top/bottom/left/right`) computed from the real
direction cosines.

Verified label sets for an axial phantom:

- axial — top `A`, bottom `P`, left `R`, right `L`
- coronal — top `S`, bottom `I`, left `R`, right `L`
- sagittal — top `S`, bottom `I`, left `A`, right `P`

Because labels come from geometry, oblique or reversed acquisitions still get
correct annotations.

`extract_mask_plane(...)` does the same for a segmentation mask so the UI can
overlay it slice-by-slice.

## 8. Slice delivery

| Endpoint | Payload |
| --- | --- |
| `/slice` | base64 raw `int16`/`float32` pixels + metadata (`rows`, `columns`, `dtype`, intensity range, window hints, labels) |
| `/thumbnail` | dependency-free 8-bit grayscale **PNG** (built with `zlib` + `struct`, no image library needed) |
| `/segmentation/{id}/mask` | bit-packed (`np.packbits`) base64 mask + shape |

For CT the slice response also carries `window_center` / `window_width` when the
series defines them; otherwise the frontend derives a robust window from the
slice's 1st/99th percentile so a few hot voxels cannot wash the image out.

## 9. Preprocessing

`preprocessing.py` provides the transforms applied before segmentation/meshing:

- intensity normalisation (recorded as `is_normalized`),
- edge-preserving smoothing for the mesh path,
- **voxel-count guards**: above `max_voxels_soft` the in-plane resolution is
  reduced before segmentation so the prototype stays interactive; above
  `max_voxels_hard` the request is refused rather than thrashing. Any reduction
  is reported.

## 10. Reliability reporting

Reconstruction never fails silently. Missing slices, irregular spacing,
orientation inconsistencies, the instance-number fallback, resampling, and
dtype downcasting each produce a `Finding` (`error`/`warning`/`info`) that
appears in the UI status bar, the operation log, and the validation report.

## 11. Known caveat: border capping

When meshing, the volume is padded so surfaces that meet the acquisition border
are closed (watertight). This is desirable for 3D printing but introduces an
artificial flat cap if the structure genuinely extends outside the field of
view. **The voxel volume is therefore the authoritative measurement**, and the
mesh volume is reported alongside it as a cross-check.
