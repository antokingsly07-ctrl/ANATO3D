"""Ad-hoc smoke test for segmentation -> mesh -> validation -> export."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app.services.analysis.measurements import measurement_summary  # noqa: E402
from app.services.dicom import reader, series as series_service  # noqa: E402
from app.services.export import service as export_service  # noqa: E402
from app.services.mesh.service import generate_mesh  # noqa: E402
from app.services.reconstruction import volume as volume_mod  # noqa: E402
from app.services.segmentation.pipeline import run_segmentation  # noqa: E402
from app.services.segmentation.registry import list_engines, list_presets  # noqa: E402
from app.services.synthetic import PhantomOptions, generate_phantom_dataset  # noqa: E402
from app.services.validation import service as validation_service  # noqa: E402


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="anato3d_seg_"))
    options = PhantomOptions(rows=128, columns=128, slices=64, slice_spacing_mm=2.0)
    phantom = generate_phantom_dataset(tmp, options)
    print(f"phantom: {len(phantom.files)} slices, expected bone volume {phantom.bone_volume_mm3 / 1000:.3f} cm3")

    dataset = reader.scan_dataset(tmp)
    analysis = series_service.analyze_dataset(dataset, tmp)
    primary = analysis.series[0]
    recon = volume_mod.build_volume(dataset, tmp, primary)
    volume = recon.volume
    print(f"volume {volume.shape} spacing {volume.spacing}")

    for engine in list_engines(volume):
        print(f"  engine {engine['key']}: available={engine['availability']['available']} "
              f"reason={engine['availability']['reason']}")

    presets = {p["key"]: p for p in list_presets(volume.modality)}
    print(f"  presets: {[k for k, p in presets.items() if p['applicable']]}")

    bone = presets["ct_bone"]["params"]
    t0 = time.time()
    segmentation = run_segmentation(volume, "threshold", dict(bone))
    print(f"segmentation (threshold bone) in {time.time() - t0:.2f}s -> {segmentation.params['statistics']}")

    # ground truth from the phantom generator
    packed = np.load(phantom.ground_truth_path)["mask"]
    gt = np.unpackbits(packed)[: int(np.prod(volume.shape))].astype(bool).reshape(volume.shape)
    report = validation_service.run_validation(
        dataset=dataset, analysis=analysis, series=primary.info, ordering=recon.ordering,
        volume=volume, segmentation=segmentation, reference_mask=gt, reference_source="phantom generator",
    )
    print(f"validation summary: {report['summary']}")
    for m in report["ground_truth"]["metrics"]:
        print(f"    {m['label']}: {m['value']} {m['unit']}")

    t0 = time.time()
    mesh = generate_mesh(segmentation, volume, output_dir=tmp / "meshes")
    print(f"mesh in {time.time() - t0:.2f}s: {mesh.stats['vertex_count']} verts / {mesh.stats['face_count']} faces")
    print(f"    files: {sorted(mesh.files)}")
    for item in measurement_summary(mesh.stats["measurements"]):
        print(f"    {item['label']}: {item['value']} {item['unit']}")
    print(f"    refinement: {mesh.stats['refinement']}")

    glb = Path(mesh.files["glb"])
    print(f"    glb size: {glb.stat().st_size / 1024:.1f} KiB")

    bundle = export_service.build_bundle(
        tmp / "bundle.zip", volume=volume, segmentation=segmentation, mesh=mesh,
        reports={"validation": report, "analysis": mesh.stats},
    )
    print(f"    bundle: {bundle.stat().st_size / 1024:.1f} KiB")

    nii = export_service.export_volume_nifti(volume, tmp / "volume.nii.gz")
    print(f"    nifti: {nii.stat().st_size / 1024:.1f} KiB affine=\n{np.round(export_service.volume_affine(volume), 3)}")

    # region growing
    rg = presets["ct_region_grow_bone"]["params"]
    t0 = time.time()
    seg2 = run_segmentation(volume, "region_growing", dict(rg))
    print(f"segmentation (region growing) in {time.time() - t0:.2f}s -> {seg2.params['statistics']}")

    # mesh from region growing, with decimation
    from app.services.mesh.refine import MeshRefinementOptions

    mesh2 = generate_mesh(seg2, volume, output_dir=tmp / "meshes2", options=MeshRefinementOptions(max_faces=20000))
    print(f"mesh2: {mesh2.stats['face_count']} faces, decimated={mesh2.stats['refinement']['decimated']}")

    # cross-method agreement (honest, not "accuracy")
    from app.services.validation.metrics import compute_metrics

    cross = compute_metrics(seg2.mask, segmentation.mask, volume.spacing)
    print("cross-method agreement threshold-vs-region-growing:")
    for m in cross.metrics[:5]:
        print(f"    {m.label}: {m.value}")

    print("\nSEGMENTATION/MESH SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
