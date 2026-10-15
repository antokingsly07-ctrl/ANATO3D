# DICOM processing

This document describes how ANATO3D ingests DICOM data, groups it, and — most
importantly — derives the correct spatial ordering.

## 1. Ingest

`POST /api/dicom/upload` accepts:

- one or more individual files,
- a `.zip` archive containing DICOM files (nested archives too).

Processing steps:

1. **Size checks** (per file and per request) happen before and during writing,
   so a lying `Content-Length` cannot exhaust memory.
2. **Filename sanitisation** (`core.security.sanitize_filename`) strips any
   directory component, normalises unicode, removes control characters, and
   refuses executable/script extensions (`validate_upload_name`).
3. **Archive extraction** is path-traversal safe and bounded.
4. Each surviving file is written under an **opaque generated name** inside the
   dataset's `original/` directory.
5. Files are then *scanned* (headers only) by `services/dicom/reader.py`:

   - `read_dataset(path)` parses the file with `pydicom`;
   - `extract_metadata(ds)` pulls the geometry-relevant tags;
   - the outcome per file (`is_dicom`, `error`) is recorded in a
     `DicomFileRecord`.

Non-image files and unreadable files are **counted and reported**, never
silently dropped:

```json
{
  "file_count": 130,
  "dicom_file_count": 128,
  "non_image_file_count": 1,
  "unreadable_file_count": 1,
  "warnings": ["…"]
}
```

## 2. Series grouping

`POST /api/dicom/analyze` runs header analysis (no pixels are read) and produces:

- **studies** (`StudySummary`) — grouped by `StudyInstanceUID`;
- **series** (`SeriesSummary`) — grouped by `SeriesInstanceUID`;
- **ordering_preview** — the first slice projections for the primary series;
- **findings** — `error` / `warning` / `info`.

For each series ANATO3D extracts:

| Field | DICOM source |
| --- | --- |
| `modality` | `Modality` |
| `rows`, `columns` | `Rows`, `Columns` |
| `orientation` | derived from `ImageOrientationPatient` (AXIAL / CORONAL / SAGITTAL / OBLIQUE) |
| `slice_spacing_mm` | median of successive projections |
| `irregular_spacing` | spacing deviation test |
| `has_complete_geometry` | every slice has orientation + position |
| `is_reconstructable` | complete geometry, ≥2 slices, consistent in-plane grid |

A series is only marked reconstructable when the geometry genuinely supports it.

## 3. Spatial ordering (the important part)

Ordering is never based on filenames or `InstanceNumber`.

### 3.1 The maths

For each slice:

```
row_dir    = ImageOrientationPatient[0:3]      # increasing column index
column_dir = ImageOrientationPatient[3:6]      # increasing row index
slice_dir  = normalize(cross(row_dir, column_dir))
```

Each slice's `ImagePositionPatient` (`origin_i`, LPS mm) is projected onto the
common slice axis:

```
projection_i = dot(origin_i, slice_dir)
```

Slices are sorted ascending by `projection_i`. Because the axis is a unit
normal of the plane, this is exact for axis-aligned, flipped, reversed, or
oblique acquisitions and is independent of how files are named or numbered.

`OrderingResult` then reports:

| Field | Meaning |
| --- | --- |
| `method` | `"geometry"` or `"instance_number_fallback"` |
| `slice_spacing_mm` | median inter-slice distance |
| `orientation_label` | AXIAL / CORONAL / SAGITTAL / OBLIQUE |
| `missing_slices` | indices where a gap exceeds the gap threshold |
| `gaps_detected` | boolean convenience flag |
| `warnings` | human-readable issues |

### 3.2 Robustness rules

- **Incomplete slices** (missing orientation or position) are skipped with a
  warning, as long as enough slices remain.
- **All slices incomplete** → `IncompleteOrientationError` (HTTP 422). The
  message is shown to the user; the remedy hint lives in the error `detail`.
- **Duplicates** (two slices at the same projection) are reported.
- **Gaps** — the largest spacing is compared with the median; a large deviation
  is reported as a probable missing slice, not silently interpolated.
- **Orientation consistency** — all slices must share the same
  `ImageOrientationPatient` within tolerance, otherwise a warning is raised.
- **Obliquity** — the slice axis is compared with the patient axes; oblique
  stacks are labelled.

### 3.3 The opt-in fallback

If a series truly has no usable geometry, the caller may pass:

```json
{ "allow_instance_number_fallback": true }
```

Under this mode ANATO3D orders by `InstanceNumber` **and raises a loud
warning**, flags the resulting volume as *spatially unverified*, and propagates
that flag into every downstream report. The default is `false`.

> The fallback exists so a researcher can still get *something* out of a broken
> export, while making it impossible to mistake the result for a geometrically
> sound reconstruction.

## 4. What is *not* done

- No sorting by filename.
- No trusting `InstanceNumber` when geometry exists.
- No silent gap interpolation.
- No dropping of non-image/unreadable files without reporting them.
- No reading of pixel data during the analysis pass (headers only, so analysis
  stays fast and safe on large studies).

## 5. Tag usage summary

| Tag | Used for |
| --- | --- |
| `SOPInstanceUID` | slice identity, dedup |
| `StudyInstanceUID` / `SeriesInstanceUID` | grouping |
| `Modality` | preset applicability, HU interpretation |
| `Rows` / `Columns` | in-plane grid, consistency checks |
| `ImageOrientationPatient` | `row_dir`, `column_dir`, `slice_dir` |
| `ImagePositionPatient` | projection / ordering |
| `SliceThickness` | corroborating spacing |
| `PixelSpacing` | in-plane mm spacing |
| `RescaleSlope` / `RescaleIntercept` | stored → HU conversion |
| `InstanceNumber` | *only* in the opt-in fallback |
| `PatientPosition` | orientation cross-check (HFS/FFS/…) |
