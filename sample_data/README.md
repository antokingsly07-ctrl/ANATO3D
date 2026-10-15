# Sample data

This folder intentionally contains **no imaging data**. Real DICOM must never be
committed to the repository (see `.gitignore`), and there is no synthetic dataset
checked in either — you generate that on demand, so it is always clearly
labelled.

You have two options for getting data into ANATO3D.

---

## Option A — Generate a labelled synthetic phantom (no download)

Fastest way to exercise the whole pipeline, including validation metrics.

**From the app:** Dataset panel → **Synthetic** → *Generate phantom*.

**From the API:**

```bash
curl -X POST http://127.0.0.1:8000/api/dicom/synthetic \
  -H "Content-Type: application/json" \
  -d '{
        "rows": 256,
        "columns": 256,
        "slices": 96,
        "irregular_spacing": false,
        "shuffle_filenames": true,
        "include_lesion": true
      }'
```

What you get:

- a CT-like series (air/lung/fat/soft-tissue/spine/rib/lesion materials with
  realistic Hounsfield values),
- **shuffled filenames by default** — proving that ordering ignores filenames,
- optional **irregular slice spacing** — exercising the gap/irregularity checks,
- a bundled **ground-truth `.npz` mask**, so validation reports real Dice/IoU/HD
  numbers.

> The phantom is explicitly **synthetic**. It is not a real anatomical image and
> must never be presented as patient data.

---

## Option B — Use a real public dataset (de-identified)

Recommended sources of genuinely de-identified data you are allowed to use:

| Dataset | Modality | Notes |
| --- | --- | --- |
| [TCIA](https://www.cancerimagingarchive.net/) | CT/MR/PET | The Cancer Imaging Archive; many collections |
| [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) | Chest CT | Lung nodules with annotations |
| [Medical Segmentation Decathlon](http://medicaldecathlon.com/) | CT/MR | Multiple tasks **with labels** you can convert to a reference mask |
| [NLST](https://www.cancerimagingarchive.net/collection/nlst/) | Chest CT | Low-dose screening CT |
| [Head-Neck-PET-CT](https://www.cancerimagingarchive.net/collection/head-neck-pet-ct/) | CT/PET | Includes RTSTRUCT contours |
| [AAPM Low Dose CT Grand Challenge](https://www.aapm.org/grandchallenge/lowdosect/) | CT | Low-dose CT |
| [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) | CT/MR | Reference segmentations for many organs |

**Before downloading:** confirm the licence and any data-use agreement, and
ensure the data is de-identified (remove or never obtain identifying tags).

### Using a dataset

1. Download the series to a local folder.
2. In the app, drag the files into the Dataset panel (**Upload DICOM**), or POST
   them to `/api/dicom/upload`. A `.zip` of the series also works.
3. Run **Inspect** to confirm the series was grouped and ordered correctly.

### Turning labels into a reference mask

If the dataset ships labels (e.g. MSD `labelsTr/*.nii.gz`, TotalSegmentator
output, or a TCIA RTSTRUCT), you can convert them to a boolean mask and upload
it to the Ground-truth field in the Validate panel:

```python
import numpy as np
import nibabel as nib

label = nib.load("labels.nii.gz").get_fdata()
mask = (label == 1)          # pick the label you segmented
np.savez_compressed("reference.npz", array=mask.astype(np.uint8))
```

Then upload `reference.npz` in the **Validate** panel. Accepted formats: `.npz`,
`.npy`, `.nii`/`.nii.gz`, a `.zip` of a DICOM mask series, or a DICOM mask
series.

The mask must match the reconstructed volume's orientation and shape; a shape
mismatch is resampled by nearest neighbour **and reported**.

---

## Format notes

- **DICOM**: single files or a `.zip`. Non-image and unreadable files are
  reported, not ignored.
- **Limits**: see `ANATO3D_MAX_FILE_BYTES`, `ANATO3D_MAX_UPLOAD_BYTES`,
  `ANATO3D_MAX_UPLOAD_FILES` in the README configuration table.

## Privacy reminder

Uploads are processed locally and are **never** uploaded anywhere by ANATO3D.
Working data lives under `ANATO3D_DATA_DIR` (`backend/_work` by default) and is
deleted automatically after the configured idle timeout. Use de-identified data
regardless.
