"""Ad-hoc smoke test for the ingestion -> ordering -> volume pipeline."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app.services.dicom import reader, series  # noqa: E402
from app.services.dicom.series_ordering import describe_frame  # noqa: E402
from app.services.reconstruction import slices, volume as volume_mod  # noqa: E402
from app.services.synthetic import PhantomOptions, generate_phantom_dataset  # noqa: E402


def run(orientation: str = "axial", irregular: bool = False) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="anato3d_smoke_"))
    options = PhantomOptions(
        rows=96,
        columns=96,
        slices=40,
        slice_spacing_mm=2.0,
        orientation=orientation,
        irregular_spacing=irregular,
    )
    t0 = time.time()
    result = generate_phantom_dataset(tmp, options)
    print(f"[{orientation}] wrote {len(result.files)} slices in {time.time() - t0:.2f}s -> {tmp}")

    t0 = time.time()
    dataset = reader.scan_dataset(tmp)
    print(f"  scanned {dataset.file_count} files, {len(dataset.dicom_files)} DICOM, {time.time() - t0:.2f}s")

    t0 = time.time()
    analysis = series.analyze_dataset(dataset, tmp)
    print(f"  analysed in {time.time() - t0:.2f}s; {len(analysis.series)} series")
    for entry in analysis.series:
        info = entry.info
        print(
            f"    series {info.series_instance_uid[-8:]} modality={info.modality} slices={info.slice_count} "
            f"rows={info.rows} cols={info.columns} orientation={info.orientation_label} "
            f"spacing={info.slice_spacing} irregular={info.irregular_spacing}"
        )
        for w in entry.ordering_warnings:
            print(f"      WARN: {w}")

    primary = analysis.series[0]
    t0 = time.time()
    recon = volume_mod.build_volume(dataset, tmp, primary)
    print(f"  reconstructed in {time.time() - t0:.2f}s")
    vol = recon.volume
    print(f"    shape={vol.shape} spacing={vol.spacing} orientation={vol.orientation_label}")
    print(f"    origin={np.round(vol.origin, 3)} det={recon.ordering.determinant:.3f}")
    print(f"    physical size mm={np.round(vol.physical_size_mm, 2)}  units={vol.intensity_units}")
    print(f"    frame: {describe_frame(recon.ordering)}")
    print(f"    HU range {vol.array.min():.0f}..{vol.array.max():.0f}")

    for plane in ("axial", "coronal", "sagittal"):
        view = slices.extract_plane(vol, plane, 0)
        print(
            f"    {plane}: {view.rows}x{view.columns} count={view.count} "
            f"pos={view.position_label} labels={view.labels['top']}/{view.labels['bottom']}/"
            f"{view.labels['left']}/{view.labels['right']}"
        )

    # ordering must be independent of filenames: check monotonic slice positions
    pos = recon.ordering.slice_positions_mm
    assert np.all(np.diff(pos) >= -1e-9), "slice positions are not monotonic"
    print("  OK: slice positions monotonic")


if __name__ == "__main__":
    for plane in ("axial", "coronal", "sagittal", "oblique"):
        run(plane)
    run("axial", irregular=True)
    print("\nSMOKE TEST PASSED")
