"""Validation endpoints: consistency checks and optional ground-truth metrics."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, Query, UploadFile

from ...core.errors import PayloadTooLargeError, ValidationError
from ...core.logging_ import get_logger
from ...core.security import validate_upload_name
from ...core.storage import new_id, store
from ...models.records import GroundTruthRecord, ValidationRecord
from ...schemas.common import JobEnvelope
from ...schemas.processing import GroundTruthUploadResponse, ValidationRequest
from ...services.validation import service as validation_service
from ...services.validation.ground_truth import align_to_shape, load_ground_truth
from .. import deps
from .. import presenters

logger = get_logger("api.validation")

router = APIRouter(prefix="/validation", tags=["validation"])


@router.post("/run", response_model=JobEnvelope, summary="Run validation")
def run(
    request: ValidationRequest,
    wait: bool = Query(False, description="Run inline and return the result instead of a job id."),
) -> JobEnvelope:
    dataset_record = deps.require_dataset_for(request.dataset_id)
    volume_record = deps.find_volume(request.volume_id) if request.volume_id else None
    segmentation_record = deps.find_segmentation(request.segmentation_id) if request.segmentation_id else None

    if dataset_record is None and volume_record is not None:
        dataset_record = deps.find_dataset(volume_record.dataset_id)
    if dataset_record is None and segmentation_record is not None:
        dataset_record = deps.find_dataset(segmentation_record.dataset_id)
    if dataset_record is None and volume_record is None and segmentation_record is None:
        raise ValidationError("Supply at least one of dataset_id, volume_id or segmentation_id to validate.")

    # Resolve the reference mask (if any) before dispatching.
    reference_mask = None
    reference_source = None
    if request.reference_segmentation_id:
        reference_record = deps.get_segmentation(request.reference_segmentation_id)
        if segmentation_record is not None and reference_record.result.mask.shape != segmentation_record.result.mask.shape:
            reference_mask = align_to_shape(
                _wrap(reference_record.result.mask, "reference segmentation"),
                tuple(segmentation_record.result.mask.shape),
            ).mask
        else:
            reference_mask = reference_record.result.mask
        reference_source = (
            "another segmentation in this session -- this is method-to-method AGREEMENT, not accuracy"
        )
    elif request.ground_truth_id:
        gt_record = deps.get_ground_truth(request.ground_truth_id)
        target_shape = None
        if segmentation_record is not None:
            target_shape = tuple(segmentation_record.result.mask.shape)
        elif volume_record is not None:
            target_shape = tuple(volume_record.volume.shape)
        if target_shape is None:
            raise ValidationError("A volume or segmentation is required to compare against a reference mask.")
        aligned = align_to_shape(gt_record.ground_truth, target_shape)
        reference_mask = aligned.mask
        reference_source = f"reference mask '{gt_record.filename}' ({gt_record.ground_truth.source})"

    def job(reporter):
        report = validation_service.run_validation(
            dataset=dataset_record.dataset if dataset_record else None,
            analysis=dataset_record.analysis if dataset_record else None,
            series=(
                deps.series_info_for_volume(dataset_record, volume_record)
                if dataset_record and volume_record
                else None
            ),
            ordering=volume_record.ordering if volume_record else None,
            volume=volume_record.volume if volume_record else None,
            segmentation=segmentation_record.result if segmentation_record else None,
            reference_mask=reference_mask,
            reference_source=reference_source,
            progress=reporter,
        )
        record = ValidationRecord(
            dataset_id=dataset_record.dataset_id if dataset_record else None,
            report=report,
        )
        deps.register_validation(record)
        reporter.step("done", 1.0, "Validation complete")
        return presenters.validation_response(report)

    return deps.dispatch("validation", job, wait=wait)


@router.get("/{validation_id}", summary="Retrieve a validation report")
def get_validation(validation_id: str) -> dict:
    return presenters.validation_response(deps.get_validation(validation_id).report)


@router.post(
    "/ground-truth",
    response_model=GroundTruthUploadResponse,
    summary="Upload a reference mask for accuracy metrics",
)
async def upload_ground_truth(
    dataset_id: str = Form(..., description="Dataset the reference mask belongs to."),
    file: UploadFile = File(..., description=".npz, .npy, .nii/.nii.gz, .zip or a DICOM mask."),
) -> GroundTruthUploadResponse:
    record = deps.get_dataset(dataset_id)
    workspace = store.workspace(dataset_id)
    safe = validate_upload_name(file.filename or "reference.npz")

    target_dir = workspace.derived_dir / "ground_truth"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{new_id('gtf')}__{safe}"

    size = 0
    with target.open("wb") as handle:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > deps.settings.max_upload_bytes:
                target.unlink(missing_ok=True)
                raise PayloadTooLargeError(
                    "The reference mask is larger than the configured upload limit.",
                    detail=f"Limit: {deps.settings.max_upload_bytes} bytes.",
                )
            handle.write(chunk)

    ground_truth = load_ground_truth(target)
    ground_truth_id = new_id("gt")
    deps.register_ground_truth(
        GroundTruthRecord(
            dataset_id=record.dataset_id,
            ground_truth_id=ground_truth_id,
            filename=safe,
            ground_truth=ground_truth,
        )
    )
    return GroundTruthUploadResponse(
        ground_truth_id=ground_truth_id,
        dataset_id=record.dataset_id,
        filename=safe,
        shape=[int(v) for v in ground_truth.shape],
        voxel_count=int(ground_truth.mask.sum()),
        notes=list(ground_truth.notes),
        warnings=list(ground_truth.warnings),
    )


def _wrap(mask, source: str):
    from ...services.validation.ground_truth import GroundTruth

    return GroundTruth(mask=mask, source=source, shape=tuple(mask.shape))


__all__ = ["router"]
