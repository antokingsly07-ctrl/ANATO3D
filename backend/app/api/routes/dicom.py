"""DICOM ingestion, dataset analysis and synthetic test data.

Every upload lives in an isolated per-dataset workspace.  Filenames are never
used to build paths (a generated opaque name is used instead) and uploaded
content is never executed or imported.  See :mod:`app.core.security`.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Query, UploadFile

from ...core.errors import (
    NoValidDicomSeriesError,
    PayloadTooLargeError,
    UnsupportedFileError,
    ValidationError,
)
from ...core.logging_ import get_logger
from ...core.security import validate_upload_name
from ...core.storage import new_id, store
from ...models.records import DatasetRecord, GroundTruthRecord
from ...schemas.common import JobEnvelope
from ...schemas.dicom import (
    AnalyzeRequest,
    AnalyzeResponse,
    DatasetSummary,
    SyntheticRequest,
    SyntheticResponse,
)
from ...services.dicom import nifti as nifti_module
from ...services.dicom import reader
from ...services.dicom import series as series_service
from ...services.dicom.series_ordering import describe_frame, order_series
from ...services.synthetic import PhantomOptions, generate_phantom_dataset
from ...services.validation import checks as checks_module
from ...services.validation.ground_truth import GroundTruth
from .. import deps
from .. import presenters

logger = get_logger("api.dicom")

router = APIRouter(prefix="/dicom", tags=["dicom"])

_CHUNK = 1 << 20  # 1 MiB
_MAX_PREVIEW_SLICES = 500


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=DatasetSummary, summary="Upload a DICOM dataset or NIfTI volume")
async def upload(
    files: list[UploadFile] = File(
        ...,
        description=(
            "DICOM files (a .zip archive is also accepted), or a NIfTI volume "
            "(.nii / .nii.gz) which is converted to DICOM automatically."
        ),
    ),
) -> DatasetSummary:
    """Store the uploaded files and scan them for DICOM series.

    The upload is size- and count-limited before anything is written.  A ``.zip``
    archive is extracted safely (member names are sanitised and blocked
    extensions are rejected); the archive itself is then removed.

    NIfTI volumes (``.nii`` / ``.nii.gz``) are converted into an equivalent DICOM
    CT series so the remainder of the pipeline is unchanged.
    """
    if not files:
        raise ValidationError("No files were included in the upload.")
    if len(files) > deps.settings.max_upload_files:
        raise ValidationError(
            f"Too many files in one upload ({len(files)} > {deps.settings.max_upload_files}). "
            "Split the dataset or upload a .zip archive."
        )

    workspace = store.create_workspace()
    uploads = workspace.uploads_dir
    uploads.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    written = 0
    archives: list[Path] = []

    for index, upload_file in enumerate(files):
        target, size = await _store_stream(upload_file, uploads, index, total_bytes)
        total_bytes += size
        written += 1
        if target.suffix.lower() == ".zip":
            archives.append(target)

    # Expand any archives, then remove them so scanning sees only members.
    for archive in archives:
        extracted, extracted_bytes = _extract_archive(archive, uploads, deps.settings.max_upload_files, total_bytes)
        total_bytes += extracted_bytes
        written += extracted
        archive.unlink(missing_ok=True)

    # If the user uploaded NIfTI volumes instead of a DICOM series, convert them
    # to DICOM in place so the rest of the pipeline is unchanged.
    converted_notes = _convert_nifti_volumes(uploads)
    for note in converted_notes:
        logger.info("Import: %s", note)

    dataset = reader.scan_dataset(uploads)
    dataset.dataset_id = workspace.dataset_id

    if not dataset.dicom_files:
        store.drop("datasets", workspace.dataset_id)
        raise NoValidDicomSeriesError(
            "None of the uploaded files could be parsed as DICOM.",
            detail=(
                f"Scanned {dataset.file_count} file(s). Accepted inputs are a DICOM series "
                "(slices or a .zip), or a NIfTI volume (.nii / .nii.gz) which is converted "
                "to DICOM automatically. Screenshots and PNGs are not supported."
            ),
        )

    record = DatasetRecord(
        dataset_id=workspace.dataset_id,
        dataset=dataset,
        root=uploads,
        warnings=list(dataset.warnings),
    )
    deps.register_dataset(record)
    logger.info("Stored dataset %s (%d file(s), %d DICOM)", record.dataset_id, written, len(dataset.dicom_files))
    return presenters.dataset_summary(record)


def _convert_nifti_volumes(uploads: Path) -> list[str]:
    """Convert any NIfTI volumes under ``uploads`` into DICOM series, in place.

    Returns human-readable notes describing what happened (used for logging).
    Conversion failures never abort the upload; the normal DICOM scan still runs.
    """
    sources = sorted(p for p in uploads.rglob("*") if p.is_file() and nifti_module.is_nifti(p))
    if not sources:
        return []

    notes: list[str] = []
    for source in sources:
        name = source.name.lower()
        base = source.name[: -len(".nii.gz")] if name.endswith(".nii.gz") else source.stem
        try:
            written = nifti_module.convert_nifti_series(source, source.parent / f"{base}_dicom")
        except Exception as exc:  # noqa: BLE001 - never let an import abort the upload
            notes.append(f"Could not convert {source.name}: {exc}")
            continue
        source.unlink(missing_ok=True)
        notes.append(f"Converted {source.name} into {len(written)} DICOM slice(s).")
    return notes


async def _store_stream(
    upload_file: UploadFile,
    dest_dir: Path,
    index: int,
    already_bytes: int,
) -> tuple[Path, int]:
    safe = validate_upload_name(upload_file.filename or f"upload_{index}")
    target = dest_dir / f"{index:06d}__{safe}"
    size = 0
    try:
        with target.open("wb") as handle:
            while True:
                chunk = await upload_file.read(_CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > deps.settings.max_file_bytes:
                    raise PayloadTooLargeError(
                        "One of the uploaded files is larger than the configured limit.",
                        detail=f"Limit: {deps.settings.max_file_bytes} bytes per file.",
                    )
                if already_bytes + size > deps.settings.max_upload_bytes:
                    raise PayloadTooLargeError(
                        "The upload exceeds the total size limit for one request.",
                        detail=f"Limit: {deps.settings.max_upload_bytes} bytes per upload.",
                    )
                handle.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return target, size


def _extract_archive(
    archive: Path,
    dest_dir: Path,
    file_budget: int,
    already_bytes: int,
) -> tuple[int, int]:
    """Extract a zip archive into ``dest_dir`` with strict safety limits."""
    written = 0
    total = 0
    try:
        handle = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as exc:  # pragma: no cover - defensive
        raise UnsupportedFileError(
            "The uploaded archive could not be read as a zip file.",
            detail=str(exc),
        ) from exc

    with handle:
        members = [m for m in handle.infolist() if not m.is_dir()]
        if written + len(members) > file_budget:
            raise PayloadTooLargeError(
                "The archive contains more files than the configured limit.",
                detail=f"Limit: {file_budget} files per upload.",
            )
        for offset, member in enumerate(members):
            safe = validate_upload_name(member.filename)
            target = dest_dir / f"zip{offset:06d}__{safe}"
            member_size = 0
            with handle.open(member) as source, target.open("wb") as sink:
                while True:
                    chunk = source.read(_CHUNK)
                    if not chunk:
                        break
                    member_size += len(chunk)
                    total += len(chunk)
                    if member_size > deps.settings.max_file_bytes:
                        raise PayloadTooLargeError(
                            "A file inside the archive is larger than the configured limit.",
                            detail=f"Member: {safe}",
                        )
                    if already_bytes + total > deps.settings.max_upload_bytes:
                        raise PayloadTooLargeError(
                            "The archive expands to more data than the configured upload limit.",
                            detail=f"Limit: {deps.settings.max_upload_bytes} bytes.",
                        )
                    sink.write(chunk)
            written += 1
    return written, total


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

@router.post("/analyze", response_model=JobEnvelope, summary="Analyse a scanned dataset")
def analyze(request: AnalyzeRequest, wait: bool = Query(True, description="Run inline and return the result.")) -> JobEnvelope:
    """Group slices into series and derive per-series geometry.

    Ordering is derived exclusively from ``ImageOrientationPatient`` /
    ``ImagePositionPatient``; ``InstanceNumber`` is never consulted here.
    """
    record = deps.get_dataset(request.dataset_id)
    root = record.root

    def job(reporter):
        reporter.step("metadata_analysis", 0.05, "Reading DICOM headers and grouping series")
        analysis = series_service.analyze_dataset(
            record.dataset,
            root,
            progress=lambda message: reporter.step("metadata_analysis", 0.4, message),
        )
        record.analysis = analysis
        record.non_image_files = analysis.non_image_files
        record.unreadable_files = analysis.unreadable_files
        record.warnings = list(analysis.warnings)

        findings = checks_module.check_dataset(record.dataset, analysis)
        summary = checks_module.summarise(findings)

        primary = analysis.series_by_uid(analysis.primary_series_uid) if analysis.primary_series_uid else None
        metadata_report: dict = {}
        preview: list[dict] = []
        frame_description: dict[str, str] = {}
        if primary is not None:
            findings.extend(checks_module.check_series(primary.info))
            summary = checks_module.summarise(findings)
            if primary.files:
                metadata_report = presenters.metadata_report(primary.files[0].meta)
            preview, frame_description, preview_warnings = _ordering_preview(primary)
            if preview_warnings:
                analysis.warnings.extend(preview_warnings)

        reporter.step("metadata_analysis", 0.95, "Series analysis complete")
        payload = AnalyzeResponse(
            dataset_id=record.dataset_id,
            studies=presenters.study_summaries(analysis),
            series=presenters.series_summaries(analysis),
            primary_series_uid=analysis.primary_series_uid,
            metadata_report=metadata_report,
            findings=presenters.findings_from_dicts([f.as_dict() for f in findings]),
            summary=summary,
            ordering_preview=preview,
            frame_description=frame_description,
            non_image_files=analysis.non_image_files,
            unreadable_files=analysis.unreadable_files,
            elapsed_s=round(float(analysis.elapsed_s), 3),
            warnings=list(analysis.warnings) + list(record.warnings),
        )
        return payload.model_dump(mode="json")

    return deps.dispatch("analyze", job, wait=wait)


def _ordering_preview(series_analysis) -> tuple[list[dict], dict[str, str], list[str]]:
    geometries = [f.geometry for f in series_analysis.files if f.geometry is not None]
    if not geometries:
        return [], {}, ["No slice carried usable patient-space geometry, so ordering could not be previewed."]

    name_by_id = {id(f.geometry): f.original_name for f in series_analysis.files if f.geometry is not None}
    try:
        ordering = order_series(geometries)
    except Exception as exc:  # noqa: BLE001 - surface any ordering problem as a warning
        return [], {}, [f"Slice ordering could not be previewed: {exc}"]

    entries: list[dict] = []
    for index, geometry in enumerate(ordering.ordered[:_MAX_PREVIEW_SLICES]):
        entries.append(
            {
                "index": index,
                "sop_instance_uid": geometry.sop_instance_uid,
                "instance_number": geometry.instance_number,
                "position_mm": [float(v) for v in geometry.image_position_patient],
                "projection_mm": round(float(geometry.sort_key), 6) if geometry.sort_key is not None else None,
                "file": name_by_id.get(id(geometry)),
            }
        )
    return entries, describe_frame(ordering), []


# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------

@router.post("/synthetic", response_model=SyntheticResponse, summary="Generate a synthetic test dataset")
def create_synthetic(request: SyntheticRequest) -> SyntheticResponse:
    """Create a clearly-labelled **synthetic** DICOM dataset.

    This is geometric phantom data, not a simulated medical image and not a real
    patient.  It exists so the whole pipeline can be exercised without
    downloading a public dataset; every file is tagged
    ``SeriesDescription = SYNTHETIC PHANTOM ...``.
    """
    workspace = store.create_workspace()
    uploads = workspace.uploads_dir
    staging = workspace.derived_dir / "synthetic_source"
    staging.mkdir(parents=True, exist_ok=True)

    options = PhantomOptions(
        rows=request.rows,
        columns=request.columns,
        slices=request.slices,
        row_spacing_mm=request.row_spacing_mm,
        column_spacing_mm=request.column_spacing_mm,
        slice_spacing_mm=request.slice_spacing_mm,
        orientation=request.orientation,
        oblique_degrees=request.oblique_degrees,
        irregular_spacing=request.irregular_spacing,
        include_lesion=request.include_lesion,
        shuffle_filenames=request.shuffle_filenames,
    )
    result = generate_phantom_dataset(staging, options)

    for source in result.files:
        shutil.copy2(source, uploads / Path(source).name)

    ground_truth_id = _register_synthetic_ground_truth(workspace, result)
    shutil.rmtree(staging, ignore_errors=True)

    dataset = reader.scan_dataset(uploads)
    dataset.dataset_id = workspace.dataset_id

    notes = list(result.notes)
    notes.append("This dataset is synthetic geometry generated by ANATO3D and is not clinical data.")

    record = DatasetRecord(
        dataset_id=workspace.dataset_id,
        dataset=dataset,
        root=uploads,
        is_synthetic=True,
        synthetic_notes=notes,
        warnings=["SYNTHETIC DATA: not a real patient and not a simulated medical image."],
        ground_truth_id=ground_truth_id,
    )
    deps.register_dataset(record)

    analysis = series_service.analyze_dataset(dataset, uploads)
    record.analysis = analysis
    record.non_image_files = analysis.non_image_files
    record.unreadable_files = analysis.unreadable_files

    return SyntheticResponse(
        dataset_id=record.dataset_id,
        slice_count=len(result.files),
        dimensions=[int(v) for v in result.dimensions],
        spacing_mm=[
            float(request.slice_spacing_mm),
            float(request.row_spacing_mm),
            float(request.column_spacing_mm),
        ],
        orientation=request.orientation,
        ground_truth_available=ground_truth_id is not None,
        ground_truth_id=ground_truth_id,
        bone_volume_cm3=round(float(result.bone_volume_mm3) / 1000.0, 4),
        notes=notes,
    )


def _register_synthetic_ground_truth(workspace, result) -> str | None:
    path = Path(result.ground_truth_path) if result.ground_truth_path else None
    if path is None or not path.exists():
        return None

    with np.load(path, allow_pickle=False) as data:
        packed = data["mask"]
    shape = tuple(int(v) for v in result.dimensions)
    mask = np.unpackbits(np.asarray(packed, dtype=np.uint8))[: int(np.prod(shape))].astype(bool)
    mask = mask.reshape(shape)

    ground_truth = GroundTruth(
        mask=mask,
        source="synthetic phantom generator (known geometry)",
        shape=mask.shape,
        notes=["Reference mask produced by the synthetic phantom generator."],
    )
    ground_truth_id = new_id("gt")
    deps.register_ground_truth(
        GroundTruthRecord(
            dataset_id=workspace.dataset_id,
            ground_truth_id=ground_truth_id,
            filename=path.name,
            ground_truth=ground_truth,
        )
    )
    return ground_truth_id


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@router.get("/{dataset_id}", response_model=DatasetSummary, summary="Retrieve a stored dataset summary")
def get_dataset_summary(dataset_id: str) -> DatasetSummary:
    return presenters.dataset_summary(deps.get_dataset(dataset_id))


__all__ = ["router"]
